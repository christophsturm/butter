// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Manifest-driven smoke test for every kernel iron generates.
//
// For each kernel listed in `kernels.metallib`'s manifest:
//   1. The PSO can be created from the compiled metallib (catches link
//      / function-constant / dispatch-signature regressions).
//   2. `maxTotalThreadsPerThreadgroup` is positive (catches kernels
//      that the Metal compiler refused to instantiate).
//
// Does NOT verify per-kernel numerical correctness — that lives in
//   - Butter's per-Op tests (Tests/ButterTests/Ops*Tests.swift), which
//     exercise every wrapper Butter actually calls via the Ops layer;
//   - iron's `iron bench` (MLX side-by-side for kernels under
//     `iron-std/src/mlx/`);
//   - per-family wrapper tests (Tests/IronSwiftTests/<Kernel>Tests.swift),
//     written incrementally as kernels change.  See `VectorAddTests`
//     for the shape.
//
// What this smoke catches that nothing else does: a PR that lands
// codegen / build-pipeline changes that fail to compile a specific
// kernel into a usable PSO.  PR #19 shipped 37 kernels with empty
// bodies via a `macro_rules!` regression; this smoke would have flagged
// the empty-body kernels at metallib-load time before Butter's higher-level
// tests caught the runtime symptom.

import Foundation
import Metal
import Testing

@testable import IronSwift

@Suite("Kernel manifest PSO smoke")
struct KernelManifestSmokeTests {

    /// MPP cooperative-tensor kernels (`_mpp_` / `bgemm`) lower simdgroup
    /// matmul that needs Apple GPU Family 9+ (M3+). The macos-latest runner
    /// pool is mixed M1/M3 hardware; on a Family < 9 host the Metal compiler
    /// rejects the cooperative-tensor body, so the manifest sweep skips those
    /// kernels there. They are covered by iron's own GPU correctness
    /// suite on adequate hardware. Detection mirrors `PSOCache.isMppKernel`.
    static let mppCooperativeTensorAvailable: Bool = {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return device.supportsFamily(.apple9)
    }()

    static func isMppKernel(_ name: String) -> Bool {
        name.contains("_mpp_") || name.contains("bgemm")
    }

    /// JSON shape mirrored from `iron-codegen::emit::write_manifest`.
    /// Only `name` is parsed — the rest is left flexible so manifest
    /// schema bumps don't fail the test.
    private struct ManifestEntry: Decodable {
        let name: String
    }

    private struct Manifest: Decodable {
        let kernels: [ManifestEntry]
    }

    private func loadManifest() throws -> Manifest {
        guard
            let url = Bundle.module.url(
                forResource: "manifest", withExtension: "json"
            )
        else {
            throw NSError(
                domain: "KernelManifestSmoke", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "manifest.json not bundled into IronSwift"
                ]
            )
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Manifest.self, from: data)
    }

    @Test("every kernel in the manifest compiles into a usable PSO")
    func everyKernelHasUsablePSO() throws {
        let manifest = try loadManifest()
        let lib = try IronLibrary()
        let cache = PSOCache(library: lib)

        // ≥ 80 keeps this honest: if PR #19's macro regression had been
        // caught here, the manifest would still have its kernels listed
        // even with empty bodies — but the count is also a smoke for
        // "manifest didn't accidentally truncate."
        #expect(
            manifest.kernels.count > 80,
            "manifest.json has \(manifest.kernels.count) kernels — expected the iron-std surface (~200+)"
        )

        var failures: [String] = []
        for entry in manifest.kernels {
            // Fast-skip the obviously-MPP kernels by name on Family < 9.
            if !Self.mppCooperativeTensorAvailable, Self.isMppKernel(entry.name) {
                continue
            }
            do {
                let pso = try cache.pipelineStateThrowing(for: entry.name)
                #expect(
                    pso.maxTotalThreadsPerThreadgroup > 0,
                    "kernel \(entry.name) reports zero maxTotalThreadsPerThreadgroup — Metal refused to instantiate"
                )
            } catch {
                // Cooperative-tensor (MMA) kernels are not all named `_mpp_` /
                // `bgemm` (e.g. the `_nax_` steel-GEMM cells). On a Family < 9
                // runner the Metal compiler rejects every cooperative-tensor
                // body with "unsupported deferred-static-alloca-size"; treat
                // that specific compile error as a skip on inadequate hardware
                // (these are verified upstream on M3+) rather than a failure.
                // Any other compile error is still a real failure.
                if !Self.mppCooperativeTensorAvailable,
                    "\(error)".contains("deferred-static-alloca-size")
                {
                    continue
                }
                failures.append("\(entry.name): \(error)")
            }
        }
        if !failures.isEmpty {
            Issue.record(
                Comment(
                    rawValue:
                        "PSO creation failed for \(failures.count) kernel(s):\n"
                        + failures.prefix(10).joined(separator: "\n")
                        + (failures.count > 10 ? "\n  …(+\(failures.count - 10) more)" : "")
                ))
        }
    }

    /// Sanity: two PSOs for the same kernel are the same instance — the
    /// PSO cache is doing its job and not recompiling per call.
    @Test("PSOCache returns the same instance for repeat lookups")
    func psoCacheDeduplicates() throws {
        let lib = try IronLibrary()
        let cache = PSOCache(library: lib)
        let a = try cache.pipelineStateThrowing(for: "vector_add_f32")
        let b = try cache.pipelineStateThrowing(for: "vector_add_f32")
        #expect(a === b, "PSOCache should return the same PSO instance on repeat lookups")
    }
}

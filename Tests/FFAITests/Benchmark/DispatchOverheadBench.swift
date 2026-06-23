// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//
// Swift -> Metal per-dispatch overhead, the apples-to-apples counterpart to the
// Rust `dispatch_bench`. Measures one gemv two ways:
//   - per-commit: one command buffer per op, commit + wait (matches the Rust
//     Device.dispatch model exactly).
//   - batched: many ops encoded into ONE command buffer, one commit (what the
//     Swift `Ops.gemv(on: cmd)` API enables, and where the real headroom is).
// The per-commit number isolates the Metal commit()+wait floor; if it lands
// near the Rust number, the host language is not the variable.

import Foundation
import Metal
import Testing

@testable import FFAI

@Suite("Dispatch overhead bench")
struct DispatchOverheadBench {
    @Test("gemv per-dispatch overhead: per-commit vs batched")
    func gemvDispatchOverhead() {
        let m = 2048
        let w = Tensor.empty(shape: [m, m], dtype: .f32)
        let x = Tensor.empty(shape: [m], dtype: .f32)

        // warmup (PSO build + first encode)
        for _ in 0..<20 {
            let cmd = Device.shared.makeCommandBuffer()
            _ = Ops.gemv(weight: w, input: x, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        let iters = 3000

        // per-commit: one command buffer per dispatch (matches Rust)
        var t0 = Date()
        for _ in 0..<iters {
            let cmd = Device.shared.makeCommandBuffer()
            _ = Ops.gemv(weight: w, input: x, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        let perCommitUs = Date().timeIntervalSince(t0) * 1e6 / Double(iters)

        // batched: encode N into one command buffer, single commit
        let batch = 500
        t0 = Date()
        let cmd = Device.shared.makeCommandBuffer()
        for _ in 0..<batch {
            _ = Ops.gemv(weight: w, input: x, on: cmd)
        }
        cmd.commit()
        cmd.waitUntilCompleted()
        let batchedUs = Date().timeIntervalSince(t0) * 1e6 / Double(batch)

        print("Swift -> Metal gemv (2048x2048):")
        print(String(format: "  per-commit (1 cmd buffer/op): %.1f us/dispatch", perCommitUs))
        print(String(format: "  batched (%d ops / 1 cmd buffer): %.1f us/op", batch, batchedUs))
        print("  (Rust per-commit, resident weights, was 178 us/dispatch)")
    }
}

// Copyright 2026 Tom Turney (@TheTom)
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
// Decode-path elementwise fusions for `NemotronHMambaLayer.decode`
// (Nemotron 3.5 Lightning's 23 mamba2 mixer layers, but the layer type
// is shared by every NemotronH checkpoint — see NemotronHText.swift).
//
// A per-layer dispatch-count audit (F-128 follow-up) counted 22
// dispatches/token in the mamba mixer decode step: 2 GEMVs (in_proj/
// out_proj), conv1d, 8x grouped `ssmStep` (n_groups=8, one thread-safe
// dispatch per group — not collapsible without a wider kernel rewrite,
// left alone per scope), and 11 small elementwise dispatches. Three of
// those elementwise pairs are a producer feeding a single consumer with
// no other reader in between, so they collapse into one dispatch each
// with zero behavior change:
//
//   addSoftplus  — `dt = softplus(dt_raw + dt_bias)`        (2 -> 1)
//   mulSilu      — `y_gated = y_skip * silu(z)`              (2 -> 1)
//   addMul       — `y_skip = y_flat + d_tiled * x`           (2 -> 1)
//
// 3 dispatches/layer x 23 mamba layers = 69 fewer dispatches/token if
// all three are on; 1/layer = 23 fewer with the default gate below (see
// the CI note — this shrank from 2/layer after addSoftplus's demotion).
//
// Bit-exactness: every op this replaces reads/writes its operands at
// `T` precision (f32/f16/bf16), computing in a float accumulator (or,
// for f16 specifically, native `half` — see `mulSilu`'s doc) and
// rounding to `T` exactly once per stock kernel (see the generated
// `iron_mul_bf16.metal` / `iron_softplus_bf16.metal` / `iron_silu_bf16.
// metal` bodies). A naive fused kernel that only rounds once at the
// very end would NOT reproduce the stock chain's *intermediate* round
// (e.g. `dt_raw + dt_bias` rounding to bf16 before softplus reads it
// back). Each kernel below explicitly re-inserts that intermediate
// `T(...)` round-trip, matching the documented contract in
// `NemotronLightningNvfp4Fuse.swift` / `LagunaGatedOProjFuse.swift`.
//
// Only `addMul` is verified bit-identical to its unfused chain across
// f32/f16/bf16 on BOTH this M5 Max and CI's runner
// (`NemotronHMambaMixerFuseTests`) — it's the sole fusion default ON,
// gated by `BUTTER_NEMOTRON_MIXER_FUSE` (`=0` restores the original
// mul+add chain byte-for-byte). Pure mul+add, no transcendentals — the
// class of op that's actually safe to assume portable.
//
// `addSoftplus` and `mulSilu` are IMPLEMENTED + TESTED but stay default
// OFF, gated separately behind their own opt-in flags
// (`BUTTER_NEMOTRON_MIXER_FUSE_SOFTPLUS=1` /
// `BUTTER_NEMOTRON_MIXER_FUSE_SILU=1`), because neither is portably
// bit-exact — both hit the same class of GPU transcendental-unit
// hazard, just discovered at different times:
//
//   - `addSoftplus` was bit-exact on this M5 Max in local development,
//     which is what shipped it default-ON in PR #128's first revision.
//     CI (a different GPU/driver) then recorded 142/512 bf16 mismatches
//     for the exact same test (`gh run view 31712976150` —
//     NemotronHMambaMixerFuseTests.swift:119). f32/f16 stayed exact on
//     CI; only bf16's extra float-promotion-then-round step exposed it.
//     The lesson: "bit-exact on the box I happened to build on" is
//     compile-context luck for an `exp`/`log`-bearing fused kernel, not
//     a property of the source — never re-promote a transcendental
//     fusion to default-ON off a single machine's test run again.
//   - `mulSilu` was never bit-exact anywhere it was tried, including
//     locally: a consistent ~20% of elements land 1-2 ULP off the
//     unfused `Ops.silu` + `Ops.mul` chain, unchanged across four
//     independently-tried source shapes (inline vs. a separate helper
//     function matching iron's own `template<T> iron_silu` shape;
//     float-cast-up vs. f16-native compute matching `iron_silu_f16`
//     exactly; `fastMathEnabled` true/false; bare `exp` vs. explicit
//     `metal::fast::exp`).
//
// Neither residual moves with any source-level or compile-flag lever
// tried, which points at the GPU's transcendental unit picking a
// context-dependent approximation (register pressure / instruction
// scheduling around the fused kernel, and apparently also GPU
// generation/driver) rather than anything controllable from here — the
// same class of non-reproducibility documented for `exp`/`log`-heavy
// fused kernels on other GPU compilers. Since the mamba SSM state is
// recurrent (carries across every decoded token), even this small a
// drift risks flipping an argmax somewhere over a long decode and
// breaking the byte-identical greedy-stream gate, so NEITHER is safe
// as a default. Both stay implemented + tested for future investigation
// (e.g. hand-written assembly-level control, or accepting a
// tolerance-gated variant) and for opt-in A/B benchmarking on hardware
// where bit-exactness has been independently re-confirmed.
//
// Kernel packaging: deliberately NOT registered under
// Sources/IronSwift/Resources/kernel_overlays/ (unlike
// `NemotronLightningNvfp4Fuse`) — these always go through `pso()`'s live
// `makeLibrary(source:)` path so each kernel can pick its OWN
// `fastMathEnabled` to match what it's replacing, measured empirically
// (see `NemotronHMambaMixerFuseTests`):
//
//   - `addMul` (pure mul + add, no transcendentals): needs
//     `fastMathEnabled = false`. With default fast-math the compiler
//     silently folds the separate `a*b` then `+base` into one hardware
//     FMA (extra internal precision, one rounding instead of two) —
//     bit-different from the two-dispatch `Ops.mul` + `Ops.add` chain.
//     Same fix as `LagunaFusedSlidingAttnStep.pso`.
//   - `addSoftplus`: default fast-math (`fastMathEnabled = true`, not
//     overridden) — its formula has no multiply to reassociate, and
//     iron's own `iron_softplus_*` (the reference) is compiled by `iron
//     build --emit` under Metal's default fast-math too, which picks a
//     faster/lower-accuracy `exp`/`log` than fast-math-off's strict-IEEE
//     ones. Matched empirically on this M5 Max — but NOT on CI's GPU
//     (see the "Bit-exactness" section above for the CI run), which is
//     exactly why this compile-mode choice alone isn't sufficient and
//     the fusion stays opt-in-only regardless of which mode is picked.
//   - `mulSilu` needs BOTH at once: `fastMathEnabled = false` (silu's
//     `zf / (1 + exp(-zf))` immediately feeding a multiply is exactly
//     the division-into-multiply pattern fast-math's algebraic
//     reassociation loves to rewrite, changing the rounding vs. the
//     unfused silu-dispatch-then-mul-dispatch chain) AND an explicit
//     `metal::fast::exp` call (fastMathEnabled=false's bare `exp` would
//     otherwise resolve to the precise/slow variant, no longer matching
//     `iron_silu_*`'s default-fast-math `exp`). Confirmed by trial: pure
//     `fastMathEnabled=true` OR pure `=false` with bare `exp` each leave
//     ~15-35% of elements off by 1-2 ULP; only the combination is exact.

import Foundation
import Metal
import IronSwift

enum NemotronHMambaMixerFuse {
    /// Gates `addMul` only — the one fusion verified bit-identical to
    /// its unfused chain on BOTH the M5 Max this was developed on AND
    /// CI's runner (pure mul+add, no transcendentals — see the file
    /// header). Default ON. `BUTTER_NEMOTRON_MIXER_FUSE=0` falls back
    /// to the original mul+add chain.
    static let enabled: Bool =
        ButterEnv["BUTTER_NEMOTRON_MIXER_FUSE"] != "0"

    /// Gates `addSoftplus` separately — bit-exact on the M5 Max this was
    /// developed on, but CI (a different GPU/driver) recorded 142/512
    /// bf16 mismatches for the SAME test (`gh run view 31712976150`).
    /// That means the "bit-exact" result was this box's compile-context
    /// luck, not a property of the kernel — the exact class of hazard
    /// already documented for `mulSilu` below. Default OFF. Opt in with
    /// `BUTTER_NEMOTRON_MIXER_FUSE_SOFTPLUS=1` for A/B benchmarking on
    /// hardware where it's been independently re-verified; do not flip
    /// the default without CI (or the target box) proving bit-exactness
    /// itself.
    static let softplusEnabled: Bool =
        ButterEnv["BUTTER_NEMOTRON_MIXER_FUSE_SOFTPLUS"] == "1"

    /// Gates `mulSilu` separately — NOT verified bit-identical anywhere
    /// (see the file header), so default OFF. Opt in with
    /// `BUTTER_NEMOTRON_MIXER_FUSE_SILU=1` for A/B benchmarking only;
    /// do not flip the default without re-establishing bit-exactness.
    static let siluEnabled: Bool =
        ButterEnv["BUTTER_NEMOTRON_MIXER_FUSE_SILU"] == "1"

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    /// `fastMath` picks the compile mode that reproduces the unfused
    /// reference chain bit-for-bit for THIS kernel — see the file
    /// header's "Kernel packaging" note for why it differs per fusion.
    private static func pso(
        _ name: String, source: String, device: MTLDevice, fastMath: Bool
    ) -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        let cacheKey = fastMath ? name : "\(name)#strict"
        if let p = psoCache[cacheKey] { return p }
        let opts = MTLCompileOptions()
        opts.fastMathEnabled = fastMath
        let lib = try! device.makeLibrary(source: source, options: opts)
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[cacheKey] = p
        return p
    }

    private static func dispatch1D(
        _ p: MTLComputePipelineState, n: Int,
        buffers: [(MTLBuffer, Int)],
        on cmd: MTLCommandBuffer
    ) {
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(p)
        for (i, (buf, off)) in buffers.enumerated() {
            enc.setBuffer(buf, offset: off, index: i)
        }
        let tg = MTLSize(width: min(256, n), height: 1, depth: 1)
        let grid = MTLSize(width: n, height: 1, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    // MARK: - `dt = softplus(dt_raw + dt_bias)`

    private static let addSoftplusSrc = """
    #include <metal_stdlib>
    using namespace metal;

    // Numerically-stable softplus, same form as `iron_softplus_*`:
    // max(x,0) + log(1 + exp(-|x|)).
    static inline float nemotron_mixer_softplus_f32(float x) {
        float zero = 0.0f;
        bool pos = x > zero;
        float ax = pos ? x : (zero - x);
        float lg = log(1.0f + exp(zero - ax));
        return (pos ? x : zero) + lg;
    }

    kernel void nemotron_mixer_add_softplus_f32(
        device const float *dt_raw [[buffer(0)]],
        device const float *dt_bias [[buffer(1)]],
        device float *out [[buffer(2)]],
        uint idx [[thread_position_in_grid]]
    ) {
        float sum = float(dt_raw[idx]) + float(dt_bias[idx]);
        float dt_sum = float(sum);  // round-trip through T=float (no-op)
        out[idx] = nemotron_mixer_softplus_f32(float(dt_sum));
    }

    kernel void nemotron_mixer_add_softplus_f16(
        device const half *dt_raw [[buffer(0)]],
        device const half *dt_bias [[buffer(1)]],
        device half *out [[buffer(2)]],
        uint idx [[thread_position_in_grid]]
    ) {
        float sum = float(dt_raw[idx]) + float(dt_bias[idx]);
        half dt_sum = half(sum);  // matches vector_add_f16's output round
        out[idx] = half(nemotron_mixer_softplus_f32(float(dt_sum)));
    }

    kernel void nemotron_mixer_add_softplus_bf16(
        device const bfloat *dt_raw [[buffer(0)]],
        device const bfloat *dt_bias [[buffer(1)]],
        device bfloat *out [[buffer(2)]],
        uint idx [[thread_position_in_grid]]
    ) {
        float sum = float(dt_raw[idx]) + float(dt_bias[idx]);
        bfloat dt_sum = bfloat(sum);  // matches vector_add_bf16's output round
        out[idx] = bfloat(nemotron_mixer_softplus_f32(float(dt_sum)));
    }
    """

    /// Fuses `Ops.add(dtRaw, dtBias)` + `Ops.softplus(dtSum)` into one
    /// dispatch. `dtRaw` and `dtBias` must share shape/dtype.
    static func addSoftplus(
        _ dtRaw: Tensor, _ dtBias: Tensor,
        on cmd: MTLCommandBuffer, device: Device,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(dtRaw.shape == dtBias.shape, "addSoftplus: shape mismatch")
        precondition(dtRaw.dtype == dtBias.dtype, "addSoftplus: dtype mismatch")
        let result = out ?? Tensor.empty(shape: dtRaw.shape, dtype: dtRaw.dtype)
        let n = dtRaw.elementCount
        let name: String
        switch dtRaw.dtype {
        case .f32: name = "nemotron_mixer_add_softplus_f32"
        case .f16: name = "nemotron_mixer_add_softplus_f16"
        case .bf16: name = "nemotron_mixer_add_softplus_bf16"
        default: fatalError("NemotronHMambaMixerFuse.addSoftplus: unsupported dtype \(dtRaw.dtype)")
        }
        let p = pso(name, source: addSoftplusSrc, device: device.mtlDevice, fastMath: true)
        dispatch1D(
            p, n: n,
            buffers: [
                (dtRaw.buffer, dtRaw.offset), (dtBias.buffer, dtBias.offset),
                (result.buffer, result.offset),
            ],
            on: cmd)
        return result
    }

    // MARK: - `y_gated = y_skip * silu(z)`

    private static let mulSiluSrc = """
    #include <metal_stdlib>
    using namespace metal;

    // Same shape as iron's own `template<typename T> inline T iron_silu(T x)`
    // helper (see `iron_silu_bf16.metal` / `iron_silu_f16.metal`) — kept as
    // a standalone function, NOT inlined directly into the kernel body.
    // Empirically load-bearing: writing the formula inline in the kernel
    // let the compiler fold it into the surrounding multiply differently
    // than when `exp`/the division sit behind their own function-call
    // boundary, which silently stopped matching iron_silu's bit pattern
    // even with fastMathEnabled=false and an explicit `metal::fast::exp`.
    // `metal::fast::exp` (not bare `exp`): this file compiles with
    // fastMathEnabled=false (see the file header's "Kernel packaging"
    // note), but iron's own `iron_silu_*` is compiled under Metal's
    // default fast-math, which resolves bare `exp` to this same
    // faster/less-precise implementation.
    template<typename T>
    static inline T nemotron_mixer_silu(T x) {
        return x / (T(1) + metal::fast::exp(-x));
    }

    kernel void nemotron_mixer_mul_silu_f32(
        device const float *a [[buffer(0)]],
        device const float *z [[buffer(1)]],
        device float *out [[buffer(2)]],
        uint idx [[thread_position_in_grid]]
    ) {
        float silu_f = nemotron_mixer_silu(float(z[idx]));
        out[idx] = float(a[idx]) * silu_f;
    }

    // f16 does NOT follow the "cast up to f32, compute, round back"
    // shape the f32/bf16 variants use — iron's own `iron_silu_f16` /
    // `iron_mul_f16` compute entirely in NATIVE half (Metal's `half` is
    // a first-class hardware type, unlike bf16 which iron promotes to
    // f32 for compute; see those kernels' generated MSL). Matching that
    // means every intermediate here (the `1+exp(-x)` denominator, the
    // division, the multiply) stays in `half`, not `float`.
    kernel void nemotron_mixer_mul_silu_f16(
        device const half *a [[buffer(0)]],
        device const half *z [[buffer(1)]],
        device half *out [[buffer(2)]],
        uint idx [[thread_position_in_grid]]
    ) {
        half silu_h = nemotron_mixer_silu(z[idx]);
        out[idx] = a[idx] * silu_h;
    }

    kernel void nemotron_mixer_mul_silu_bf16(
        device const bfloat *a [[buffer(0)]],
        device const bfloat *z [[buffer(1)]],
        device bfloat *out [[buffer(2)]],
        uint idx [[thread_position_in_grid]]
    ) {
        float silu_f = nemotron_mixer_silu(float(z[idx]));
        bfloat z_act = bfloat(silu_f);  // matches iron_silu_bf16's output round
        out[idx] = bfloat(float(a[idx]) * float(z_act));
    }
    """

    /// Fuses `Ops.silu(z)` + `Ops.mul(a, zAct)` into one dispatch. `a`
    /// and `z` must share shape/dtype.
    static func mulSilu(
        _ a: Tensor, _ z: Tensor,
        on cmd: MTLCommandBuffer, device: Device,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(a.shape == z.shape, "mulSilu: shape mismatch")
        precondition(a.dtype == z.dtype, "mulSilu: dtype mismatch")
        let result = out ?? Tensor.empty(shape: a.shape, dtype: a.dtype)
        let n = a.elementCount
        let name: String
        switch a.dtype {
        case .f32: name = "nemotron_mixer_mul_silu_f32"
        case .f16: name = "nemotron_mixer_mul_silu_f16"
        case .bf16: name = "nemotron_mixer_mul_silu_bf16"
        default: fatalError("NemotronHMambaMixerFuse.mulSilu: unsupported dtype \(a.dtype)")
        }
        let p = pso(name, source: mulSiluSrc, device: device.mtlDevice, fastMath: false)
        dispatch1D(
            p, n: n,
            buffers: [
                (a.buffer, a.offset), (z.buffer, z.offset),
                (result.buffer, result.offset),
            ],
            on: cmd)
        return result
    }

    // MARK: - `y_skip = y_flat + d_tiled * x`

    private static let addMulSrc = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void nemotron_mixer_add_mul_f32(
        device const float *base [[buffer(0)]],
        device const float *a [[buffer(1)]],
        device const float *b [[buffer(2)]],
        device float *out [[buffer(3)]],
        uint idx [[thread_position_in_grid]]
    ) {
        float prod = float(a[idx]) * float(b[idx]);
        float dx = float(prod);  // round-trip through T=float (no-op)
        out[idx] = float(base[idx]) + float(dx);
    }

    kernel void nemotron_mixer_add_mul_f16(
        device const half *base [[buffer(0)]],
        device const half *a [[buffer(1)]],
        device const half *b [[buffer(2)]],
        device half *out [[buffer(3)]],
        uint idx [[thread_position_in_grid]]
    ) {
        float prod = float(a[idx]) * float(b[idx]);
        half dx = half(prod);  // matches iron_mul_f16's output round
        out[idx] = half(float(base[idx]) + float(dx));
    }

    kernel void nemotron_mixer_add_mul_bf16(
        device const bfloat *base [[buffer(0)]],
        device const bfloat *a [[buffer(1)]],
        device const bfloat *b [[buffer(2)]],
        device bfloat *out [[buffer(3)]],
        uint idx [[thread_position_in_grid]]
    ) {
        float prod = float(a[idx]) * float(b[idx]);
        bfloat dx = bfloat(prod);  // matches iron_mul_bf16's output round
        out[idx] = bfloat(float(base[idx]) + float(dx));
    }
    """

    /// Fuses `Ops.mul(a, b)` + `Ops.add(base, dx)` into one dispatch.
    /// `base`, `a`, `b` must share shape/dtype.
    static func addMul(
        base: Tensor, _ a: Tensor, _ b: Tensor,
        on cmd: MTLCommandBuffer, device: Device,
        into out: Tensor? = nil
    ) -> Tensor {
        precondition(
            base.shape == a.shape && a.shape == b.shape, "addMul: shape mismatch")
        precondition(
            base.dtype == a.dtype && a.dtype == b.dtype, "addMul: dtype mismatch")
        let result = out ?? Tensor.empty(shape: base.shape, dtype: base.dtype)
        let n = base.elementCount
        let name: String
        switch base.dtype {
        case .f32: name = "nemotron_mixer_add_mul_f32"
        case .f16: name = "nemotron_mixer_add_mul_f16"
        case .bf16: name = "nemotron_mixer_add_mul_bf16"
        default: fatalError("NemotronHMambaMixerFuse.addMul: unsupported dtype \(base.dtype)")
        }
        let p = pso(name, source: addMulSrc, device: device.mtlDevice, fastMath: false)
        dispatch1D(
            p, n: n,
            buffers: [
                (base.buffer, base.offset), (a.buffer, a.offset), (b.buffer, b.offset),
                (result.buffer, result.offset),
            ],
            on: cmd)
        return result
    }
}

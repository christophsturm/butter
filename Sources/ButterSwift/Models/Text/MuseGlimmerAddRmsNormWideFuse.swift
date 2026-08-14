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
// glimmer-decode-gap lever 4: fused (residual `a + b`) + RMSNorm for
// hidden sizes ABOVE `OpsValidation.addRmsNormMaxRowSize` (4096) — Muse
// Glimmer's hidden_size is 6656, which blows straight past that cap, so
// `Ops.addAndRmsNorm`'s existing fast (TPG = n/4, 4-elements/thread)
// kernel is unreachable for this model and `MuseGlimmerLayer.forward`'s
// `OpsValidation.validateAddRmsNorm(n: hidden) == nil` check always
// takes the unfused fallback (separate `Ops.add` + `Ops.rmsNorm`(wide)
// dispatches). This file adds the missing "wide" fused variant so the
// postAttn-residual / pre-FF-norm pair collapses to one dispatch there
// too.
//
// **Byte-exactness design — why this is NOT `iron_add_rms_norm`'s
// shape.** `iron_add_rms_norm_{f32,f16,bf16}` (the existing narrow
// kernel, see `Ops.swift`'s `addAndRmsNorm`) computes its sum-of-squares
// and normed output from the UNROUNDED float sum `a+b`, never rounding
// the residual to `T` before using it in the norm math — a legitimate,
// slightly-more-accurate fusion, but numerically DIFFERENT from running
// `Ops.add` (which rounds `a+b` to `T` once) followed by a separate
// `Ops.rmsNorm` dispatch (which reads that already-rounded `T` value
// back and casts it back up to float). Since Muse Glimmer's shipped
// baseline — hidden=6656 always takes the unfused fallback — IS that
// two-dispatch chain, this kernel reproduces THAT rounding discipline
// instead: pass 1 computes `sum = float(a)+float(b)`, rounds it to `T`
// exactly once, writes it to `residualOut`, and accumulates
// sum-of-squares from the ROUNDED value (matching `Ops.rmsNorm`'s wide
// kernel reading a `T`-typed input and casting up). Pass 2 re-reads
// `residualOut` (each thread reads back only the indices it itself
// wrote in pass 1 — no cross-thread device-memory dependency, so the
// two `threadgroup_barrier` calls around the reduction are sufficient;
// nothing additional is needed for the residual round-trip).
//
// The two-pass strided-loop shape (TPG fixed at 1024, `tpg = n_simd *
// 32` stride, scalar accumulate-in-a-loop rather than 4-wide
// vectorized) is copied from `iron_rms_norm_wide_*` verbatim — NOT from
// `iron_add_rms_norm`'s 4-wide vectorized shape — because
// `iron_rms_norm_wide` (scalar loop) is what the unfused fallback's
// SECOND dispatch actually runs for n=6656 (`dispatchRmsNorm` routes
// n > 4096 to the wide kernel). Matching accumulation ORDER matters for
// floating-point sums (see `LagunaResidualRmsRouterFuse.swift`'s header
// for a fusion that explicitly gave up byte-parity for a different
// reason) — reusing the exact same loop shape/stride/reduction tree
// keeps the sum-of-squares reduction order identical to the reference.
//
// The residual round-to-`T`-then-read-back step is intentionally
// uniform across f32/f16/bf16 (`T(float(a) + float(b))`): promoting two
// `T`-precision operands to `float`, adding, and rounding once to `T`
// is bit-identical to a native `T`-precision add for both `half` and
// `bfloat` (float32 has more than enough headroom to represent the
// exact sum of two `half`/`bfloat` values with no intermediate
// rounding), so this single shape matches `vector_add_f16`'s NATIVE
// half add, `vector_add_bf16`'s explicit promote-add-round, and
// `vector_add_f32`'s direct float add all at once — verified by
// `MuseGlimmerAddRmsNormWideFuseTests`.
//
// **Compile flags — empirically determined, not assumed.** The kernel
// is compiled with `fastMathEnabled = true`. The first cut of this file
// shipped `fastMathEnabled = false` on the theory that a hand-written
// kernel held to a byte-exact bar should avoid the compiler
// reassociating anything — that theory was WRONG: a live 200-token
// greedy A/B against the unfused two-dispatch baseline
// (`glimmer-decode-gap` census run, 3 prompts) diverged at token 30 and
// token 166 on two of the three prompts with `fastMathEnabled = false`.
// Flipping to `true` made all three prompts byte-identical across the
// full 200 tokens ON THIS M5 MAX. Read: the precompiled
// `iron_rms_norm_wide_*` / `vector_add_*` kernels this replaces were
// themselves built under Metal's default (fast-math-on) compile mode,
// same lesson `NemotronHMambaMixerFuse.swift`'s `addSoftplus` section
// documents — "correct-looking" reasoning about compiler behavior is
// not a substitute for measuring against the actual reference kernels.
//
// **This "fix" was itself hardware-local — do not trust it either.**
// PR #137 shipped this kernel default-ON off the M5 Max result above.
// CI (a different GPU/driver) failed it: `MuseGlimmerAddRmsNormWideFuseTests`
// recorded 46-195 bitwise mismatches per dtype (`gh run view
// 31775982914`), while every one of the other 2029 tests in the run —
// including Levers 1-3, which reuse existing shipped kernels rather
// than compiling a new one — passed clean. This is the EXACT hazard
// class `NemotronHMambaMixerFuse.swift`'s `addSoftplus` hit: bit-exact
// on the machine it was developed on, then wrong on CI's GPU, because
// `rsqrt`/transcendental rounding is compile-context- and
// hardware-dependent, not a portable property of the kernel source.
// One machine's green run — even a green run that itself corrected an
// earlier wrong assumption — is still just one machine's green run.
//
// Escape hatch, now inverted to opt-in: `BUTTER_GLIMMER_WIDE_ADDNORM=1`
// enables this kernel; unset/anything else keeps the unfused `Ops.add`
// + `Ops.rmsNorm` two-dispatch chain, which is what every other
// architecture already runs at this hidden size and what CI is green
// on. Default OFF — the measured decode delta from this specific lever
// was ≈0 anyway (25.35 tok/s levers 1-3 vs 25.36 tok/s with this kernel
// also on, both within the run-to-run noise floor), so there is no
// perf case for accepting a portability risk this kernel hasn't
// cleared. Flip back to opt-in-tested/default-ON only after this
// passes CI's GPU directly, the same bar `addSoftplus` never cleared.

import Foundation
import Metal
import IronSwift

enum MuseGlimmerAddRmsNormWideFuse {
    /// Row-width floor: below this the existing fast `Ops.addAndRmsNorm`
    /// (TPG = n/4) kernel already applies and should be preferred — this
    /// wide kernel is only for hidden sizes past
    /// `OpsValidation.addRmsNormMaxRowSize`.
    static let minRowSize = OpsValidation.addRmsNormMaxRowSize + 1

    /// Opt-in, OFF by default — CI's GPU recorded 46-195 bitwise
    /// mismatches per dtype for this kernel (`gh run view
    /// 31775982914`) even though it was verified byte-identical in a
    /// live decode A/B on the M5 Max it was developed on. Same
    /// cross-GPU `rsqrt`/transcendental hazard class as
    /// `NemotronHMambaMixerFuse.swift`'s `addSoftplus`. See the file
    /// header's "This fix was itself hardware-local" section.
    /// `BUTTER_GLIMMER_WIDE_ADDNORM=1` to opt in for A/B benchmarking
    /// on hardware where bit-exactness has been independently
    /// re-verified; do not flip the default without CI proving it.
    static let enabled: Bool =
        ButterEnv["BUTTER_GLIMMER_WIDE_ADDNORM"] == "1"

    nonisolated(unsafe) private static var psoCache: [String: MTLComputePipelineState] = [:]
    private static let lock = NSLock()

    private static func pso(
        _ name: String, source: String, device: MTLDevice
    ) -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = psoCache[name] { return p }
        // fastMathEnabled = true — empirically required to match the
        // precompiled `iron_rms_norm_wide_*` / `vector_add_*` reference
        // kernels bit-for-bit on THIS box. See the file header's
        // "Compile flags" section: `false` measurably diverged in a
        // live decode A/B — and even this `true` setting only proved
        // bit-exact on the M5 Max it was tuned on, not on CI's GPU
        // (hence `enabled` defaulting OFF; see the header's "This fix
        // was itself hardware-local" section).
        let opts = MTLCompileOptions()
        opts.fastMathEnabled = true
        let lib = try! device.makeLibrary(source: source, options: opts)
        let fn = lib.makeFunction(name: name)!
        let p = try! device.makeComputePipelineState(function: fn)
        psoCache[name] = p
        return p
    }

    private static let src = """
    #include <metal_stdlib>
    using namespace metal;

    #define MUSEGLIMMER_ADD_RMS_NORM_WIDE_KERNEL(NAME, T, TOFLOAT, FROMFLOAT) \\
    kernel void NAME( \\
        const device T *a [[buffer(0)]], \\
        const device T *b [[buffer(1)]], \\
        const device T *w [[buffer(2)]], \\
        const device float *eps_buf [[buffer(3)]], \\
        device T *residual_out [[buffer(4)]], \\
        device T *normed_out [[buffer(5)]], \\
        constant uint &n [[buffer(6)]], \\
        uint3 _tid3  [[thread_position_in_threadgroup]], \\
        uint3 _tgid3 [[threadgroup_position_in_grid]], \\
        uint3 _lsize3 [[threads_per_threadgroup]], \\
        uint simd_lane [[thread_index_in_simdgroup]], \\
        uint simd_group [[simdgroup_index_in_threadgroup]] \\
    ) { \\
        uint tid    = _tid3.x; \\
        uint tgid_x = _tgid3.x; \\
        uint lsize  = _lsize3.x; \\
        uint n_simd = lsize / 32u; \\
        if (n_simd == 0u) { return; } \\
        threadgroup float tg_ssq_sg[32]; \\
        uint row = tgid_x; \\
        uint rs = row * n; \\
        uint tpg = n_simd * 32u; \\
        float acc = 0.0f; \\
        for (uint i = tid; i < n; i += tpg) { \\
            uint idx = rs + i; \\
            float sum = TOFLOAT(a[idx]) + TOFLOAT(b[idx]); \\
            T rounded = FROMFLOAT(sum); \\
            residual_out[idx] = rounded; \\
            float rf = TOFLOAT(rounded); \\
            acc = acc + (rf * rf); \\
        } \\
        float tg_ssq; \\
        { \\
            float sv = simd_sum(acc); \\
            if (simd_lane == 0) tg_ssq_sg[simd_group] = sv; \\
            threadgroup_barrier(mem_flags::mem_threadgroup); \\
            if (simd_group == 0) { \\
                float wv = simd_lane < n_simd ? tg_ssq_sg[simd_lane] : 0.0f; \\
                tg_ssq_sg[0] = simd_sum(wv); \\
            } \\
            threadgroup_barrier(mem_flags::mem_threadgroup); \\
            tg_ssq = tg_ssq_sg[0]; \\
        } \\
        float eps = eps_buf[0]; \\
        float rms = rsqrt((tg_ssq / float(n)) + eps); \\
        for (uint i = tid; i < n; i += tpg) { \\
            uint idx = rs + i; \\
            float rf = TOFLOAT(residual_out[idx]); \\
            float wi = TOFLOAT(w[i]); \\
            normed_out[idx] = FROMFLOAT((rf * rms) * wi); \\
        } \\
    }

    MUSEGLIMMER_ADD_RMS_NORM_WIDE_KERNEL(
        museglimmer_add_rms_norm_wide_f32, float, float, float)
    MUSEGLIMMER_ADD_RMS_NORM_WIDE_KERNEL(
        museglimmer_add_rms_norm_wide_f16, half, float, half)
    MUSEGLIMMER_ADD_RMS_NORM_WIDE_KERNEL(
        museglimmer_add_rms_norm_wide_bf16, bfloat, float, bfloat)
    """

    /// Fused (residual `a + b`) + RMSNorm for row widths above
    /// `OpsValidation.addRmsNormMaxRowSize` — the wide-row sibling of
    /// `Ops.addAndRmsNorm`. `a`, `b` are `[rows, n]` flat; `weight` is
    /// `[n]`. See the file header for the byte-exactness design.
    static func addAndRmsNormWide(
        _ a: Tensor, _ b: Tensor, weight: Tensor, eps: Float,
        nRows: Int, rowSize: Int, device: Device,
        on cmd: MTLCommandBuffer,
        residualOut: Tensor? = nil, normedOut: Tensor? = nil
    ) -> (residual: Tensor, normed: Tensor) {
        precondition(
            a.elementCount == nRows * rowSize,
            "MuseGlimmerAddRmsNormWideFuse: a size \(a.elementCount) ≠ nRows*rowSize")
        precondition(
            b.elementCount == nRows * rowSize,
            "MuseGlimmerAddRmsNormWideFuse: b size \(b.elementCount) ≠ nRows*rowSize")
        precondition(
            weight.elementCount == rowSize,
            "MuseGlimmerAddRmsNormWideFuse: weight must be [rowSize]")
        precondition(
            a.dtype == b.dtype && a.dtype == weight.dtype,
            "MuseGlimmerAddRmsNormWideFuse: dtype mismatch")
        precondition(
            rowSize >= minRowSize,
            "MuseGlimmerAddRmsNormWideFuse: rowSize \(rowSize) < \(minRowSize) — "
                + "use Ops.addAndRmsNorm instead")

        let resid = residualOut ?? Tensor.empty(shape: a.shape, dtype: a.dtype, device: device)
        let normed = normedOut ?? Tensor.empty(shape: a.shape, dtype: a.dtype, device: device)

        let name: String
        switch a.dtype {
        case .f32: name = "museglimmer_add_rms_norm_wide_f32"
        case .f16: name = "museglimmer_add_rms_norm_wide_f16"
        case .bf16: name = "museglimmer_add_rms_norm_wide_bf16"
        default: fatalError("MuseGlimmerAddRmsNormWideFuse: unsupported dtype \(a.dtype)")
        }
        let p = pso(name, source: src, device: device.mtlDevice)

        let epsBuf = Device.shared.scalarBuffer(eps)
        // Fixed TPG = 1024 (32 simdgroups), matching `iron_rms_norm_wide`'s
        // dispatch shape exactly — see the file header for why the
        // stride/reduction order must match.
        let tgWidth = 1024
        let grid = MTLSize(width: nRows * tgWidth, height: 1, depth: 1)
        let tg = MTLSize(width: tgWidth, height: 1, depth: 1)
        var nU = UInt32(rowSize)

        guard let enc = cmd.makeComputeCommandEncoder() else {
            return (resid, normed)
        }
        enc.setComputePipelineState(p)
        enc.setBuffer(a.buffer, offset: a.offset, index: 0)
        enc.setBuffer(b.buffer, offset: b.offset, index: 1)
        enc.setBuffer(weight.buffer, offset: weight.offset, index: 2)
        enc.setBuffer(epsBuf, offset: 0, index: 3)
        enc.setBuffer(resid.buffer, offset: resid.offset, index: 4)
        enc.setBuffer(normed.buffer, offset: normed.offset, index: 5)
        enc.setBytes(&nU, length: 4, index: 6)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()

        return (resid, normed)
    }
}

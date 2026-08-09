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
// RouterSetsDebug: per-token, per-MoE-layer top-K expert selection dump.
//
// `BUTTER_DEBUG_ROUTER_SETS=1` prints one line per MoE layer per decode
// token to stderr: the selected top-K expert ids and an approximate
// "router logit" (the sigmoid(raw GEMV)+bias score the selection kernel
// compared candidates on) for each — format
// `[router] t=<i> layer=<l> experts=[id:logit,...]`.
//
// This exists to settle butter#108: does Laguna's GPU-chained decode
// loop (`forward(tokenIdBuffer:)`, the default greedy-argmax fast path
// via `GpuTokenLoop`) select a DIFFERENT expert set than the host loop
// (`forward(tokenId:)`, the `BUTTER_GREEDY_GPU_ARGMAX=0` cpuSample path)
// for the same prompt/position — the hypothesis being a near-tied
// top-8-of-256 router decision flips under nvfp4 quantization noise. See
// `research/retrieval_attention` / the W43 laguna-drift investigation and
// `LogitMarginDebug.swift` (the sibling instrument this one is modeled
// on) for the argmax-level half of the same question.
//
// Wiring: `MoELayer` stashes the raw router logits + bias tensors it
// used for selection (see `debugRouterSnapshot()` in MoELayer.swift, and
// its call sites in `runPathBRouteChain` / `decodeGPUPathBFull` /
// `decodeGPUPathBHybrid` — every place Laguna's top-8 router chain runs,
// NVFP4 stacked-expert and affine-4bit alike). `LagunaModel.
// forwardFromTokenTensor` (shared body for BOTH `forward(tokenId:)` and
// `forward(tokenIdBuffer:)` — the whole point is comparing them) takes a
// debug-only fully-synchronous per-layer branch when this flag is on,
// so each layer's router-selection scratch tensors are read back before
// the NEXT layer/token can overwrite them (Laguna's decode scratches are
// grow-only, reused in place across calls — an async readback here would
// race a later layer/token's write under the default nonblock decode
// window). Off, this branch is never taken.
//
// Cost: guarded by `enabled` at every call site (one `ProcessInfo`
// dictionary lookup cached implicitly, same discipline as
// `LogitMarginDebug`), so near-zero when off. When on, it collapses
// Laguna's whole pipe/nonblock command-buffer batching to a synchronous
// commit+wait per layer (~40x/token instead of ~1x/token) — expected and
// fine, this is a debug flag, not a perf-preserving one.

import Foundation

public enum RouterSetsDebug {
    public static var enabled: Bool {
        ButterEnv["BUTTER_DEBUG_ROUTER_SETS"] == "1"
    }

    /// Current generated-token index, 1-based (t=0 is the prefill-seed
    /// token in `LogitMarginDebug`'s convention and never reaches a
    /// per-token `decode()` call, so the first token that DOES — the
    /// first `forwardFromTokenTensor` invocation of a generation — is
    /// t=1, matching butter#108's "divergence is at t=1").
    nonisolated(unsafe) private static var tokenCounter: Int = 0
    private static let lock = NSLock()

    /// Resets the per-generation token counter. Call once per
    /// `driveGeneration` invocation, mirroring `LogitMarginDebug.reset()`.
    public static func reset() {
        lock.lock()
        tokenCounter = 0
        lock.unlock()
    }

    /// Advances to the next decode token. Call once per
    /// `forwardFromTokenTensor` invocation (i.e. once per generated
    /// token, NOT per layer) — every MoE layer's report during that call
    /// shares the resulting token index.
    public static func nextToken() {
        lock.lock()
        tokenCounter += 1
        lock.unlock()
    }

    private static var currentToken: Int {
        lock.lock()
        defer { lock.unlock() }
        return tokenCounter
    }

    /// Reports the top-K selected experts for one MoE layer at the
    /// current token. `indices` are the selected top-K expert ids and
    /// `scores` their per-expert `sigmoid(logit)+bias` router score
    /// (already recomputed by the caller — see
    /// `MoELayer.debugRouterSnapshot()`), the same value the stock
    /// four-dispatch chain and `LagunaRouterTop8Fuse`'s fused kernel both
    /// compare candidates on (see that file's header) — a diagnostic for
    /// how close the selection was, NOT a bit-exact replication of the
    /// kernel's own intermediate bf16 rounding (this is a debug print,
    /// not a correctness oracle). No-op unless `enabled`.
    public static func report(layer: Int, indices: [Int], scores: [Float]) {
        guard enabled else { return }
        let t = currentToken
        let parts = zip(indices, scores).map { id, score in "\(id):\(score)" }
        fputs("[router] t=\(t) layer=\(layer) experts=[\(parts.joined(separator: ","))]\n", stderr)
    }
}

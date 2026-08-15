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
// Mamba2LayerCache — per-layer state bundle for a Mamba 2 mixer block.
// Wraps the two caches a Mamba 2 layer needs:
//   - `ssm`  : fp32 recurrent state h[nHeads, stateDim, headDim]
//              (selective-scan compresses arbitrary history into this)
//   - `conv` : rolling window of the last `(kernelSize - 1)` post-projection
//              inputs, shape [kernelSize - 1, nChannels], in layer dtype
//
// Conforms to `LayerCacheProtocol` (not `KVCacheProtocol`) — Mamba 2 has
// no K / V tensors and the engine never calls attention methods on these
// caches. The Mamba 2 model.forward casts each per-layer cache back to
// `Mamba2LayerCache` to reach the typed members.
//
// Length semantics: `length` increments by 1 per decode step purely for
// accounting; the underlying storage is constant-size. `maxSeq` is
// reported as `.max` so external code that gates on capacity treats SSM
// layers as unlimited.

import Foundation
import Metal

public final class Mamba2LayerCache: LayerCacheProtocol, @unchecked Sendable {
    public let ssm: SSMStateCache
    public let conv: ConvStateCache

    public private(set) var length: Int = 0
    public let maxSeq: Int = .max

    public init(
        nHeads: Int, stateDim: Int, headDim: Int,
        convChannels: Int, convKernelSize: Int,
        dtype: DType, device: Device = .shared
    ) {
        self.ssm = SSMStateCache(
            nHeads: nHeads, stateDim: stateDim,
            headDim: headDim, device: device)
        self.conv = ConvStateCache(
            nChannels: convChannels,
            kernelSize: convKernelSize,
            dtype: dtype, device: device)
    }

    public func reset() {
        ssm.reset()
        conv.reset()
        length = 0
    }

    /// Called by the Mamba 2 layer after each decode step to advance the
    /// step counter. The actual state mutation happens inside the SSM /
    /// conv kernels via the wrapped caches.
    public func advance() { length += 1 }

    /// Batched-prefill counterpart to `advance()` — advances the step
    /// counter by `n` in one call instead of looping `advance()` `n`
    /// times. Used by `NemotronHMambaLayer.decodeMulti` (F-13x chunked
    /// scan), which processes `n` rows per call without ever calling
    /// the per-token `decode(...)`.
    public func advanceBy(_ n: Int) { length += n }

    public var bytesAllocated: Int {
        ssm.bytesAllocated + conv.bytesAllocated
    }

    public var bytesInUse: Int {
        // SSM + conv storage is constant-size; "in use" equals allocated
        // once any step has executed, zero before then.
        length == 0 ? 0 : bytesAllocated
    }

    /// Set the step counter directly without touching `ssm`/`conv`
    /// storage. Mirrors `GDNStateCache.setLength` — the spec-decode
    /// restore path writes the snapshot tensors into `ssm.h` /
    /// `conv.state` itself (via `restoreEncode`) and then calls this to
    /// resync the accounting counter, without the zero-fill `reset()`
    /// would do.
    public func setLength(_ length: Int) {
        precondition(length >= 0, "Mamba2LayerCache.setLength: must be ≥ 0")
        self.length = length
    }

    /// Composite content-level snapshot: both `ssm.h` and `conv.state`,
    /// queued on the caller's shared command buffer (see
    /// `CacheSnapshot.snapshotAll`'s per-layer batching rationale).
    public func snapshotEncode(on cmd: MTLCommandBuffer, device: Device = .shared)
        -> (ssm: Tensor, conv: Tensor)
    {
        (ssm: ssm.snapshotEncode(on: cmd, device: device), conv: conv.snapshotEncode(on: cmd, device: device))
    }

    /// Composite content-level restore; see `snapshotEncode`. Does NOT
    /// touch `length` — callers pair this with `setLength(_:)`, same
    /// two-call contract `CacheSnapshot.restoreAll` uses for
    /// `GDNStateCache`/`Qwen35GDNLayerCache`.
    public func restoreEncode(ssm: Tensor, conv: Tensor, on cmd: MTLCommandBuffer) {
        self.ssm.restoreEncode(from: ssm, on: cmd)
        self.conv.restoreEncode(from: conv, on: cmd)
    }
}

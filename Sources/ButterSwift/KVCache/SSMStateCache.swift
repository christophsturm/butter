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
// SSMStateCache — per-layer recurrent state for Mamba 2 (and other
// state-space-model layers). Unlike `KVCache`, the state has a fixed
// shape (`[nHeads, stateDim, headDim]`) and doesn't grow with the
// number of generated tokens — selective scan compresses an
// arbitrary-length history into that fixed state. Decode is therefore
// O(1) memory and O(stateDim * headDim) compute per token.
//
// Storage dtype is **fp32**: the state accumulates `exp(A*dt) * h +
// dt*B*x` over many decode steps; bf16's 7-bit mantissa drifts in a
// few dozen steps. mlx-swift-lm makes the same call.
//
// This class ships the storage + reset/step plumbing. The Mamba 2
// family file calls `Ops.ssmStep(...)` between
// `appendOnGPU(...)`-equivalent invocations; the conv-state buffer
// that hybrid models need (Mamba 2 has a 1D depthwise conv on
// the input projection) is a follow-up.

import Foundation
import Metal

public final class SSMStateCache: @unchecked Sendable {
    public let nHeads: Int
    public let stateDim: Int
    public let headDim: Int

    /// Recurrent state h, shape `[nHeads, stateDim, headDim]`. Always fp32.
    public let h: Tensor

    public init(
        nHeads: Int, stateDim: Int, headDim: Int,
        device: Device = .shared
    ) {
        self.nHeads = nHeads
        self.stateDim = stateDim
        self.headDim = headDim
        self.h = Tensor.empty(
            shape: [nHeads, stateDim, headDim],
            dtype: .f32, device: device)
        self.h.zero()
    }

    /// Reset the recurrent state to zero. Cheap (zero-fill of a small
    /// fp32 buffer); cache size doesn't depend on sequence length.
    public func reset() { h.zero() }

    /// Bytes occupied by the recurrent state. Independent of how many
    /// tokens have been processed — that's the SSM compression win.
    public var bytesAllocated: Int {
        nHeads * stateDim * headDim * DType.f32.byteSize
    }

    /// Cached snapshot tensor. Reused across snapshot calls, mirroring
    /// `ConvStateCache.snapshotEncode`'s scratch-reuse discipline —
    /// avoids churning `nHeads*stateDim*headDim*4` bytes per spec-decode
    /// verify cycle.
    private var cachedSnapshot: Tensor?

    /// Encode-only content snapshot: queues the blit on `cmd` and
    /// returns the destination WITHOUT committing. Mirrors
    /// `ConvStateCache.snapshotEncode` / `GDNStateCache.snapshotEncode`
    /// so a composite (`Mamba2LayerCache`) — and, above it,
    /// `CacheSnapshot.snapshotAll` — can batch every layer's SSM +
    /// conv blit onto ONE command buffer instead of one round-trip per
    /// layer per cache.
    ///
    /// Unlike `ConvStateCache` (fixed length-only cost per
    /// `Mamba2LayerCache`), the SSM state is the recurrent Mamba 2
    /// scan's accumulator (`h`) — content-level (not length-only)
    /// restore is required because, unlike attention KV, a rejected
    /// speculative token's contribution to `h` cannot be undone by
    /// truncating a growing buffer; the state must be blitted back to
    /// its pre-step value.
    public func snapshotEncode(on cmd: MTLCommandBuffer, device: Device = .shared) -> Tensor {
        let shape = [nHeads, stateDim, headDim]
        if cachedSnapshot == nil {
            cachedSnapshot = Tensor.empty(shape: shape, dtype: .f32, device: device)
        }
        let snap = cachedSnapshot!
        guard let blit = cmd.makeBlitCommandEncoder() else {
            preconditionFailure("SSMStateCache.snapshotEncode: makeBlitCommandEncoder failed")
        }
        blit.copy(
            from: h.buffer, sourceOffset: h.offset,
            to: snap.buffer, destinationOffset: snap.offset, size: bytesAllocated)
        blit.endEncoding()
        return snap
    }

    /// Encode-only restore; see `snapshotEncode`.
    public func restoreEncode(from snapshot: Tensor, on cmd: MTLCommandBuffer) {
        precondition(
            snapshot.elementCount == h.elementCount,
            "SSMStateCache.restoreEncode: snapshot has \(snapshot.elementCount) elements, "
                + "expected \(h.elementCount)")
        guard let blit = cmd.makeBlitCommandEncoder() else {
            preconditionFailure("SSMStateCache.restoreEncode: makeBlitCommandEncoder failed")
        }
        blit.copy(
            from: snapshot.buffer, sourceOffset: snapshot.offset,
            to: h.buffer, destinationOffset: h.offset, size: bytesAllocated)
        blit.endEncoding()
    }

    /// Committing snapshot/restore pair — convenience wrappers around
    /// the encode-only variants for callers outside a shared
    /// command-buffer batch (e.g. unit tests). Each commits and waits.
    public func snapshot(device: Device = .shared) -> Tensor {
        let cmd = device.makeCommandBuffer()
        let snap = snapshotEncode(on: cmd, device: device)
        cmd.commit()
        cmd.waitUntilCompleted()
        return snap
    }

    public func restore(from snapshot: Tensor, device: Device = .shared) {
        let cmd = device.makeCommandBuffer()
        restoreEncode(from: snapshot, on: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
    }
}

extension Array where Element == SSMStateCache {
    /// Sum of `bytesAllocated` across all per-layer SSM caches.
    public var totalBytesAllocated: Int { reduce(0) { $0 + $1.bytesAllocated } }
}

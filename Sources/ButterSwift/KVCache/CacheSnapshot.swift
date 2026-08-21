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
// CacheSnapshot — composite snapshot/restore over `[any LayerCacheProtocol]`.
//
// Speculative decode needs to roll back ALL of a model's per-layer
// caches on draft rejection: KVCache (attention), GDNStateCache +
// ConvStateCache (GDN). This file gives the spec-decode driver one
// call to snapshot every layer + one call to restore.
//
// Per-cache-kind snapshot payload:
//   * KVCache              → `length` integer only (the KV slots
//                            beyond `length` are unused; appending
//                            after restore overwrites them).
//   * GDNStateCache        → Tensor snapshot of `current` +
//                            `length` integer.
//   * ConvStateCache       → Tensor snapshot of `state`.
//   * Qwen35GDNLayerCache  → both of the above (conv + GDN +
//                            shared `length`).
//
// Cost on Qwen3.6-A3B: 60 MiB GDN state snapshots + 900 KiB conv
// snapshots + ~zero attention metadata = ~61 MiB per spec step. On
// Apple silicon unified memory at ~400 GB/s that's ~150 µs in
// aggregate blit time — negligible vs ~60 ms decode step.

import Foundation

/// Per-layer snapshot. The enum cases match the concrete cache types
/// the model layer slots take; the spec-decode driver routes per
/// layer.
public enum LayerCacheSnapshot {
    case kv(length: Int, absolutePosition: Int)
    case gdn(currentState: Tensor, length: Int)
    case conv(state: Tensor)
    case gdnLayer(conv: Tensor, gdnState: Tensor, length: Int)
    /// `Mamba2LayerCache` (NemotronH's Mamba 2 mixer): content-level
    /// snapshot of BOTH the SSM recurrent state and the conv rolling
    /// window, plus the step-counter `length`. Recurrent state can't be
    /// rolled back by length truncation the way KV can — restoring the
    /// actual tensor content is the only correct rollback for a
    /// rejected speculative token (see `SSMStateCache.snapshotEncode`'s
    /// doc).
    case mamba2(ssm: Tensor, conv: Tensor, length: Int)
    /// `StatelessLayerCache` (a pure MLP/MoE feed-forward layer — no
    /// per-token state at all). Nothing to snapshot or restore; this
    /// case exists purely so a heterogeneous hybrid stack with
    /// stateless layer slots (e.g. NemotronH's MLP/MoE "-"/"E" layers)
    /// doesn't hit `snapshotAll`'s catch-all `preconditionFailure`.
    case stateless
}

extension Array where Element == any LayerCacheProtocol {
    var supportsSnapshotAll: Bool {
        allSatisfy { cache in
            if let kv = cache as? any KVCacheProtocol {
                if case .unbounded = kv.eviction { return true }
                return false
            }
            return cache is Qwen35GDNLayerCache || cache is GDNStateCache
                || cache is Mamba2LayerCache || cache is StatelessLayerCache
        }
    }

    /// Snapshot every layer cache. The blits themselves are ~150 µs
    /// aggregate; the cost that matters is SYNCHRONISATION, so every
    /// layer's blit is encoded onto ONE command buffer and committed
    /// once. The per-cache `snapshot()` helpers each commit and wait,
    /// which at 48 GDN layers (conv + state apiece) meant ~96 blocking
    /// round-trips per spec-decode cycle - measured at ~25 ms, i.e. most
    /// of the snapshot cost, versus ~1 ms batched.
    public func snapshotAll(
        device: Device = .shared,
        waitUntilCompleted: Bool = true
    ) -> [LayerCacheSnapshot] {
        let cmd = device.makeCommandBuffer()
        let out = self.map { cache -> LayerCacheSnapshot in
            if let composite = cache as? Qwen35GDNLayerCache {
                return .gdnLayer(
                    conv: composite.conv.snapshotEncode(on: cmd, device: device),
                    gdnState: composite.gdn.snapshotEncode(on: cmd, device: device),
                    length: composite.length)
            }
            if let gdn = cache as? GDNStateCache {
                return .gdn(
                    currentState: gdn.snapshotEncode(on: cmd, device: device),
                    length: gdn.length)
            }
            if let kv = cache as? any KVCacheProtocol {
                return .kv(
                    length: kv.length,
                    absolutePosition: kv.absolutePosition)
            }
            if let mamba = cache as? Mamba2LayerCache {
                let snap = mamba.snapshotEncode(on: cmd, device: device)
                return .mamba2(ssm: snap.ssm, conv: snap.conv, length: mamba.length)
            }
            if cache is StatelessLayerCache {
                return .stateless
            }
            preconditionFailure(
                "CacheSnapshot: unhandled LayerCacheProtocol subtype \(type(of: cache)). Add a case if a new cache kind ships."
            )
        }
        cmd.commit()
        if waitUntilCompleted { cmd.waitUntilCompleted() }
        return out
    }

    /// Restore every layer cache from a snapshot taken via `snapshotAll`.
    /// Caller must pass the same `[any LayerCacheProtocol]` instance
    /// the snapshot came from — order + identity must match.
    /// Restore every cache from `snapshots`.
    ///
    /// `recurrentOnly` restores just the conv/GDN recurrent state and
    /// leaves KV caches alone. Speculative rollback needs this: the
    /// batched verify pass is causal, so the KV entries it wrote for
    /// ACCEPTED positions are already correct and must be kept, and the
    /// caller shortens the KV length to the accepted count afterwards.
    /// Restoring KV here would truncate back to the pre-verify length,
    /// and `truncate` cannot extend it again.
    public func restoreAll(
        from snapshots: [LayerCacheSnapshot],
        device: Device = .shared,
        recurrentOnly: Bool = false
    ) {
        precondition(
            self.count == snapshots.count,
            "CacheSnapshot.restoreAll: layer count mismatch (caches=\(self.count), snapshots=\(snapshots.count))"
        )
        let cmd = device.makeCommandBuffer()
        for (cache, snap) in zip(self, snapshots) {
            switch (cache, snap) {
            case let (composite as Qwen35GDNLayerCache, .gdnLayer(convT, gdnT, length)):
                composite.conv.restoreEncode(from: convT, on: cmd)
                composite.gdn.restoreEncode(from: gdnT, on: cmd)
                composite.setLength(length)
            case let (gdn as GDNStateCache, .gdn(currentT, length)):
                gdn.restoreEncode(from: currentT, on: cmd)
                gdn.setLength(length)
            case let (kv as any KVCacheProtocol, .kv(length, _)):
                if !recurrentOnly { kv.truncate(toLength: length) }
            case let (mamba as Mamba2LayerCache, .mamba2(ssmT, convT, length)):
                mamba.restoreEncode(ssm: ssmT, conv: convT, on: cmd)
                mamba.setLength(length)
            case (is StatelessLayerCache, .stateless):
                break
            default:
                preconditionFailure(
                    "CacheSnapshot.restoreAll: cache/snapshot mismatch at layer — cache=\(type(of: cache)), snapshot=\(snap)"
                )
            }
        }
        cmd.commit()
        cmd.waitUntilCompleted()
    }
}

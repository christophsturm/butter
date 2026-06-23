// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
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
// Device — singleton wrapper over the system MTLDevice + a default
// command queue. Exposes a single Sendable handle that all of FFAI uses
// to allocate buffers and submit work.

import Foundation
import Metal
import MetalTileSwift

public final class Device: @unchecked Sendable {
    public let mtlDevice: MTLDevice
    public let commandQueue: MTLCommandQueue

    /// Lazy MTLResidencySet that pins the model's weight buffers so
    /// every command buffer skips per-allocation residency tracking.
    /// Populated after `Model.load` finishes. Typed as `Any?` so the
    /// deployment target stays below macOS 15; the cast back to
    /// `MTLResidencySet` lives inside the `@available` block in
    /// `markWeightsResident`. Initialised under `residencyLock` to
    /// single-flight the descriptor build.
    private var weightResidencySet: Any?
    private let residencyLock = NSLock()

    public static let shared: Device = {
        // Reuse the same MTLDevice + queue MetalTileSwift uses, so PSOs
        // and buffers are guaranteed compatible.
        let lib = MetalTileLibrary.shared
        return Device(mtlDevice: lib.device, commandQueue: lib.commandQueue)
    }()

    public init(mtlDevice: MTLDevice, commandQueue: MTLCommandQueue) {
        self.mtlDevice = mtlDevice
        self.commandQueue = commandQueue
    }

    // ─── Scratch slab — generic transient-buffer allocator ────────────
    //
    // `Device.makeBuffer` is the default path for persistent buffers.
    // For transients that live for the duration of a forward sub-block
    // — and would otherwise hammer Metal's internal driver pool with
    // hundreds of `makeBuffer(length:)` calls per token — there's a
    // **scratch slab**: a single pre-allocated `MTLBuffer` that callers
    // slice into via offset bumps. `device.allocScratch(bytes:)` returns
    // `(buffer, offset)`; `Tensor.scratch(shape:dtype:)` wraps the slice
    // as a Tensor; `device.resetScratch()` rewinds the offset to 0.
    //
    // Wrap a sub-block in `device.withScratch { ... }`: it flips
    // `scratchModeActive` on (so plain `Tensor.empty` routes through the
    // slab) and rewinds the offset at scope exit. State that CARRIES
    // OVER between scratch scopes (e.g., the mHC 4-channel residual)
    // must NOT live in scratch — allocate it with the default
    // `Device.makeBuffer` instead.
    public var scratchSlabBytes: Int = 256 * 1024 * 1024  // 256 MB cap
    private var scratchBuffer: MTLBuffer?
    private var scratchOffset: Int = 0

    /// When `true`, `Tensor.empty(...)` routes through the scratch slab
    /// instead of allocating a fresh MTLBuffer. Set by
    /// `withScratch { ... }` so callers don't need to switch every
    /// allocation site over to `Tensor.scratch` explicitly.
    public var scratchModeActive: Bool = false

    // ─── Allocation counters (diagnostic) ────────────────────────────
    public var bufferAllocCount: Int = 0
    public var bufferAllocBytes: Int = 0
    public var scratchAllocCount: Int = 0
    public var scratchAllocBytes: Int = 0

    // ─── Dequant-intermediate scratch (persistent reusable buffer) ────
    //
    // GGUF dequant kernels need 1-2 large transient buffers per call
    // (e.g., IQ2_XXS expert tensor: ~524 MB qs intermediate + ~32 MB
    // d_f32 scales). Caller commits + waits the dequant cmd buffer
    // BEFORE returning, so the intermediate is safely reusable
    // across calls. These slabs grow lazily to the largest size
    // requested.
    private var dequantIntermediateBuffers: [String: MTLBuffer] = [:]
    private let scratchLock = NSLock()

    /// Returns a pre-allocated MTLBuffer ≥ `minBytes` keyed by `tag`.
    /// Thread-safe: multiple parallel staging tasks may call with
    /// distinct slot-keyed tags concurrently.
    public func intermediateScratch(tag: String, minBytes: Int) -> MTLBuffer {
        scratchLock.lock()
        defer { scratchLock.unlock() }
        let need = max(minBytes, 64)
        if let buf = dequantIntermediateBuffers[tag], buf.length >= need {
            return buf
        }
        let alloc = max(need, (dequantIntermediateBuffers[tag]?.length ?? 0) * 2)
        guard let buf = mtlDevice.makeBuffer(length: alloc, options: .storageModeShared) else {
            fatalError("Device.intermediateScratch: failed to allocate \(alloc)-byte slab")
        }
        dequantIntermediateBuffers[tag] = buf
        return buf
    }

    /// Process RSS in KB via a `ps` shell-out. Slow (~10 ms per call)
    /// but works without entitlements. Use sparingly — only at
    /// per-sub-block instrumentation points.
    public static func currentRssKB() -> Int {
        let pid = ProcessInfo.processInfo.processIdentifier
        let task = Process()
        task.launchPath = "/bin/ps"
        task.arguments = ["-o", "rss=", "-p", "\(pid)"]
        let pipe = Pipe()
        task.standardOutput = pipe
        do { try task.run() } catch { return -1 }
        task.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let s =
            String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? "0"
        return Int(s) ?? 0
    }

    /// Allocate `bytes` from the scratch slab (lazily creating the slab
    /// on first use). 16-byte aligned. Fatal if the slab overflows —
    /// caller should size `scratchSlabBytes` to fit one sub-block of
    /// transients.
    public func allocScratch(bytes: Int) -> (buffer: MTLBuffer, offset: Int) {
        if scratchBuffer == nil {
            scratchBuffer = mtlDevice.makeBuffer(
                length: scratchSlabBytes, options: .storageModeShared)
            guard scratchBuffer != nil else {
                fatalError("Device.allocScratch: failed to allocate \(scratchSlabBytes)-byte slab")
            }
        }
        let aligned = (scratchOffset + 15) & ~15
        if aligned + bytes > scratchSlabBytes {
            fatalError(
                "Device.allocScratch: slab overflow — needed \(aligned + bytes), have \(scratchSlabBytes). Caller should resetScratch() between sub-blocks or grow scratchSlabBytes."
            )
        }
        scratchOffset = aligned + bytes
        scratchAllocCount += 1
        scratchAllocBytes += bytes
        return (scratchBuffer!, aligned)
    }

    /// Reset the scratch slab offset to 0. **Every Tensor sliced into
    /// the slab via `Tensor.scratch(...)` becomes invalid after this
    /// call** — all sub-block-local transients must be done with.
    public func resetScratch() {
        scratchOffset = 0
    }

    /// Convenience scope wrapper — runs the body with
    /// `scratchModeActive = true` (so `Tensor.empty` transparently
    /// uses the scratch slab), then resets the slab at scope exit.
    /// Any Tensor sliced into the slab inside the body is INVALID
    /// once `body` returns — carry-over state must be copied to a
    /// persistent buffer (or allocated via `Tensor.empty` while
    /// `scratchModeActive == false`) before the scope exits.
    public func withScratch<T>(_ body: () throws -> T) rethrows -> T {
        let wasActive = scratchModeActive
        scratchModeActive = true
        defer {
            if !wasActive {
                scratchModeActive = false
                resetScratch()
            }
        }
        return try body()
    }

    /// Allocate a fresh shared-storage MTLBuffer of the given byte length.
    public func makeBuffer(length: Int) -> MTLBuffer {
        guard let buf = mtlDevice.makeBuffer(length: length, options: .storageModeShared) else {
            fatalError("Device.makeBuffer(length: \(length)) returned nil")
        }
        bufferAllocCount += 1
        bufferAllocBytes += length
        return buf
    }

    /// Ensure the scratch slab is at least `bytes`, reallocating if needed.
    /// SAFE ONLY when no scratch slices are live (`scratchOffset == 0`) —
    /// call at the top of a forward pass before any `allocScratch`. The slab
    /// is a single reused buffer (not a per-call allocation), so growing it
    /// for a large prefill chunk is bounded, not a leak. Decode keeps 256 MB.
    public func ensureScratchSlab(_ bytes: Int) {
        if let buf = scratchBuffer, buf.length >= bytes { return }
        precondition(
            scratchOffset == 0,
            "ensureScratchSlab: cannot resize with \(scratchOffset) bytes of live slices")
        scratchSlabBytes = bytes
        scratchBuffer = mtlDevice.makeBuffer(length: bytes, options: .storageModeShared)
        guard scratchBuffer != nil else {
            fatalError("ensureScratchSlab: failed to allocate \(bytes)-byte slab")
        }
    }

    // Cache of 4-byte scalar-argument buffers, keyed by value. Kernel
    // scalar args (rmsNorm eps, RoPE start/step, …) were allocating a
    // fresh 4-byte MTLBuffer on EVERY op call — ~5 rmsNorms/layer ×
    // 43 layers = ~220 tiny allocations per token. Over a long
    // (e.g. 32k) decode that churned millions of buffers and eventually
    // tripped `makeBuffer returned nil`. Scalars are ~constant, so cache
    // one reusable buffer per value.
    nonisolated(unsafe) private var scalarBufCache: [Float: MTLBuffer] = [:]
    private let scalarBufLock = NSLock()
    public func scalarBuffer(_ value: Float) -> MTLBuffer {
        scalarBufLock.lock()
        defer { scalarBufLock.unlock() }
        if let b = scalarBufCache[value] { return b }
        guard let b = mtlDevice.makeBuffer(length: 4, options: .storageModeShared) else {
            fatalError("Device.scalarBuffer: makeBuffer(4) returned nil")
        }
        var v = value
        memcpy(b.contents(), &v, 4)
        scalarBufCache[value] = b
        bufferAllocCount += 1
        bufferAllocBytes += 4
        return b
    }

    /// Make a new MTLCommandBuffer.
    public func makeCommandBuffer() -> MTLCommandBuffer {
        guard let cb = commandQueue.makeCommandBuffer() else {
            fatalError("Device.makeCommandBuffer() returned nil")
        }
        return cb
    }

    /// Add `buffers` to a persistent MTLResidencySet attached to the
    /// command queue. Without this, Apple's Metal driver re-validates
    /// per-allocation residency on every command-buffer encode — at
    /// model sizes with tens of thousands of dispatches per prefill,
    /// the per-dispatch overhead dominates wall time. One residency
    /// set is shared across all weight buffers; repeated calls add
    /// to it. Requires macOS 15+ / iOS 18+; older OSes silently
    /// no-op. Set `FFAI_NO_RESIDENCY_SET=1` to disable for A/B.
    public func markWeightsResident(_ buffers: [MTLBuffer]) {
        if ProcessInfo.processInfo.environment["FFAI_NO_RESIDENCY_SET"] != nil { return }
        guard #available(macOS 15.0, iOS 18.0, *) else { return }
        residencyLock.lock()
        defer { residencyLock.unlock() }
        if weightResidencySet == nil {
            let descriptor = MTLResidencySetDescriptor()
            descriptor.label = "FFAI weights"
            descriptor.initialCapacity = max(buffers.count, 1024)
            do {
                let set = try mtlDevice.makeResidencySet(descriptor: descriptor)
                commandQueue.addResidencySet(set)
                weightResidencySet = set
            } catch {
                // Driver refused to create the set; fall back to default
                // residency tracking. Not fatal — just slower.
                weightResidencySet = nil
                return
            }
        }
        guard let set = weightResidencySet as? MTLResidencySet else { return }
        for buf in buffers {
            set.addAllocation(buf)
        }
        set.commit()
        set.requestResidency()
    }
}

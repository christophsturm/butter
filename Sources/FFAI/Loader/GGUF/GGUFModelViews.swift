// Copyright 2026 Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//
// Zero-copy GPU-resident weight views over the GGUF tensor-data region.
//
// Wrap the tensor-data region in a HANDFUL of overlapping page-aligned
// `newBufferWithBytesNoCopy` views. Each view is <= maxBufferLength;
// adjacent views overlap by (maxTensorBytes + page) so EVERY tensor lies
// wholly inside one view — a hot kernel takes one `(MTLBuffer, innerOffset)`.
//
// CRITICAL: this wraps the reader's EXISTING mmap (GGUFReader.mmapBase) —
// it does NOT create a second mapping. A second mmap of the 86 GB file
// would double resident memory (the no-copy views pin every page they
// touch), causing pageouts. Views hold no-copy buffers over pages the
// reader already owns; we never munmap here.

import Foundation
import Metal

#if canImport(Darwin)
import Darwin
#endif

/// A set of overlapping no-copy MTLBuffer views over the reader's mmap.
public final class GGUFModelViews {
    public struct View {
        public let buffer: MTLBuffer
        public let fileOffset: Int   // absolute file byte offset this view starts at
        public let length: Int
    }

    public let views: [View]

    /// Wrap the reader's EXISTING mmap (no second mapping) in overlapping
    /// no-copy GPU views.
    /// - Parameters:
    ///   - mmapBase: the reader's stable, page-aligned mmap base pointer.
    ///   - fileSize: mapped byte count.
    ///   - dataStart: absolute file offset where tensor data begins.
    ///   - maxTensorBytes: largest tensor byte length (sets view overlap).
    public init?(mmapBase: UnsafeRawPointer, fileSize: Int, dataStart: Int,
                 maxTensorBytes: Int, device: Device) {
        let page = Int(getpagesize())
        guard UInt(bitPattern: mmapBase) & UInt(page - 1) == 0 else { return nil }  // page-aligned

        // Page-align the cover start down to a page boundary at/below dataStart.
        let coverStart = dataStart & ~(page - 1)
        let coverLen = fileSize - coverStart
        guard coverLen > 0, maxTensorBytes > 0 else { return nil }
        let regionBase = UnsafeMutableRawPointer(mutating: mmapBase.advanced(by: coverStart))

        // Cap views at <4 GiB so kernels can address any inner byte offset
        // with u32. The overlap invariant keeps every tensor wholly inside a
        // view, so the max in-view index is <= viewSize < 2^32. (Capping is
        // simpler than a u64 offset + the full maxBufferLength.)
        let maxBuffer = Swift.min(device.mtlDevice.maxBufferLength, 4_000_000_000) & ~(page - 1)
        if maxBuffer == 0 { return nil }

        // Overlap so every tensor fits wholly in one view (overlap invariant).
        let overlap = ((maxTensorBytes + page - 1) & ~(page - 1)) + page
        guard maxBuffer > overlap else { return nil }
        let step = maxBuffer - overlap

        var built: [View] = []
        var off = 0
        while off < coverLen {
            let viewBytes = Swift.min(maxBuffer, coverLen - off)
            guard let buf = device.mtlDevice.makeBuffer(
                bytesNoCopy: regionBase.advanced(by: off),
                length: viewBytes,
                options: [.storageModeShared],
                deallocator: nil)  // reader owns the mmap; we never unmap
            else { return nil }
            buf.label = "gguf_model_view_\(built.count)"
            built.append(View(buffer: buf, fileOffset: coverStart + off, length: viewBytes))
            if viewBytes == coverLen - off { break }
            off += step
        }
        self.views = built
    }

    /// Return the view (buffer + inner byte offset) that wholly contains
    /// the file byte range `[absStart, absStart+length)`.
    public func view(absStart: Int, length: Int) -> (buffer: MTLBuffer, offset: Int)? {
        for v in views {
            if absStart >= v.fileOffset && absStart + length <= v.fileOffset + v.length {
                return (v.buffer, absStart - v.fileOffset)
            }
        }
        return nil
    }
}

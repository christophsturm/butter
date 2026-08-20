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

import Metal
import Testing

@testable import ButterSwift

@Suite("AURA batched encode parity")
struct AURABatchedEncodeParityTests {
    @Test("cache scatter preserves encoded rows")
    func cacheScatterPreservesRows() {
        let device = Device.shared
        let rows = 4
        let dim = 256
        let maxSeq = 8
        let scheme = AURAScheme.aura4v2
        let rotation = AURAQuantizedKVCacheRotations.build(
            headDim: dim, layerIndex: 3, activationDtype: .bf16, device: device)
        let kCodebook = AURACodebook.centroidsTensor(
            dim: dim, bits: scheme.keyBits, dtype: .bf16, device: device)
        let kBoundaries = AURACodebook.boundariesTensor(
            dim: dim, bits: scheme.keyBits, dtype: .bf16, device: device)
        let vCodebook = AURACodebook.centroidsTensor(
            dim: dim, bits: scheme.valueBits, dtype: .bf16, device: device)
        let vBoundaries = AURACodebook.boundariesTensor(
            dim: dim, bits: scheme.valueBits, dtype: .bf16, device: device)
        let cache = AURAQuantizedKVCache(
            nKVHeads: rows, headDim: dim, maxSeq: maxSeq, dtype: .bf16,
            scheme: scheme,
            rotation: rotation.rotation, rotationT: rotation.rotationT,
            rotationDtype: rotation.rotationDtype, rotationDtypeT: rotation.rotationDtypeT,
            kCodebook: kCodebook, kBoundaries: kBoundaries,
            vCodebook: vCodebook, vBoundaries: vBoundaries,
            sharedWorkingK: Tensor.empty(shape: [rows, maxSeq, dim], dtype: .bf16),
            sharedWorkingV: Tensor.empty(shape: [rows, maxSeq, dim], dtype: .bf16),
            decodePath: .dequantMirror, device: device)

        let appendedPositions = 4
        let inputs = (0 ..< appendedPositions).map { position in
            var values = [UInt16](repeating: 0, count: rows * dim)
            for i in values.indices {
                let phase = Float(position) * 0.37
                let value =
                    sin(Float(i) * 0.071 + phase) * 0.4
                    + cos(Float(i) * 0.019 - phase) * 0.2
                values[i] = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
            }
            let input = Tensor.empty(shape: [rows, dim], dtype: .bf16, device: device)
            input.copyIn(from: values)
            return input
        }
        let expectedPacked = Tensor.empty(
            shape: [appendedPositions, rows, cache.kPackedWidth], dtype: .u32, device: device)
        let expectedNorms = Tensor.empty(
            shape: [appendedPositions, rows], dtype: .bf16, device: device)
        let expectedVPacked = Tensor.empty(
            shape: [appendedPositions, rows, cache.vPackedWidth], dtype: .u32, device: device)
        let expectedVNorms = Tensor.empty(
            shape: [appendedPositions, rows], dtype: .bf16, device: device)
        expectedPacked.zero()
        expectedNorms.zero()
        expectedVPacked.zero()
        expectedVNorms.zero()
        let expectedCommand = device.makeCommandBuffer()
        for position in 0 ..< appendedPositions {
            for row in 0 ..< rows {
                Ops.auraEncode(
                    input: Tensor(
                        buffer: inputs[position].buffer,
                        offset: inputs[position].offset + row * dim * 2,
                        shape: [1, dim], dtype: .bf16),
                    rotation: rotation.rotationDtype,
                    boundaries: kBoundaries, codebook: kCodebook,
                    packedOut: Tensor(
                        buffer: expectedPacked.buffer,
                        offset: expectedPacked.offset
                            + (position * rows + row) * cache.kPackedWidth * 4,
                        shape: [1, cache.kPackedWidth], dtype: .u32),
                    normsOut: Tensor(
                        buffer: expectedNorms.buffer,
                        offset: expectedNorms.offset + (position * rows + row) * 2,
                        shape: [1], dtype: .bf16),
                    rows: 1, dim: dim, packedWidth: cache.kPackedWidth,
                    bits: scheme.keyBits, on: expectedCommand)
                Ops.auraEncode(
                    input: Tensor(
                        buffer: inputs[position].buffer,
                        offset: inputs[position].offset + row * dim * 2,
                        shape: [1, dim], dtype: .bf16),
                    rotation: rotation.rotationDtype,
                    boundaries: vBoundaries, codebook: vCodebook,
                    packedOut: Tensor(
                        buffer: expectedVPacked.buffer,
                        offset: expectedVPacked.offset
                            + (position * rows + row) * cache.vPackedWidth * 4,
                        shape: [1, cache.vPackedWidth], dtype: .u32),
                    normsOut: Tensor(
                        buffer: expectedVNorms.buffer,
                        offset: expectedVNorms.offset + (position * rows + row) * 2,
                        shape: [1], dtype: .bf16),
                    rows: 1, dim: dim, packedWidth: cache.vPackedWidth,
                    bits: scheme.valueBits, on: expectedCommand)
            }
        }
        expectedCommand.commit()
        expectedCommand.waitUntilCompleted()

        let cacheCommand = device.makeCommandBuffer()
        for input in inputs {
            cache.appendOnGPU(kFlat: input, vFlat: input, on: cacheCommand)
        }
        cacheCommand.commit()
        cacheCommand.waitUntilCompleted()

        let expectedPackedValues = expectedPacked.toArray(as: UInt32.self)
        let cachePackedValues = cache.kPacked.toArray(as: UInt32.self)
        let expectedNormValues = expectedNorms.toArray(as: UInt16.self)
        let cacheNormValues = cache.kNorms.toArray(as: UInt16.self)
        let expectedVPackedValues = expectedVPacked.toArray(as: UInt32.self)
        let cacheVPackedValues = cache.vPacked.toArray(as: UInt32.self)
        let expectedVNormValues = expectedVNorms.toArray(as: UInt16.self)
        let cacheVNormValues = cache.vNorms.toArray(as: UInt16.self)
        for row in 0 ..< rows {
            for position in 0 ..< appendedPositions {
                let expectedStart = (position * rows + row) * cache.kPackedWidth
                let cacheStart = (row * maxSeq + position) * cache.kPackedWidth
                #expect(
                    Array(
                        expectedPackedValues[
                            expectedStart ..< expectedStart + cache.kPackedWidth])
                        == Array(cachePackedValues[cacheStart ..< cacheStart + cache.kPackedWidth]))
                #expect(
                    expectedNormValues[position * rows + row]
                        == cacheNormValues[row * maxSeq + position])
                let expectedVStart = (position * rows + row) * cache.vPackedWidth
                let cacheVStart = (row * maxSeq + position) * cache.vPackedWidth
                #expect(
                    Array(
                        expectedVPackedValues[
                            expectedVStart ..< expectedVStart + cache.vPackedWidth])
                        == Array(
                            cacheVPackedValues[
                                cacheVStart ..< cacheVStart + cache.vPackedWidth]))
                #expect(
                    expectedVNormValues[position * rows + row]
                        == cacheVNormValues[row * maxSeq + position])
            }
        }
    }

    @Test("four bf16 rows match four single-row dispatches")
    func fourRowsMatchSingleRows() {
        let device = Device.shared
        let rows = 4
        let dim = 256
        let bits = 4
        let packedWidth = AURACodebook.packedWidth(dim: dim, bits: bits)

        var values = [UInt16](repeating: 0, count: rows * dim)
        for i in values.indices {
            let value = sin(Float(i) * 0.071) * 0.4 + cos(Float(i) * 0.019) * 0.2
            values[i] = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
        }
        let input = Tensor.empty(shape: [rows, dim], dtype: .bf16, device: device)
        input.copyIn(from: values)

        let identity = AURARotation.identityMatrix(dim: dim)
        let identityBits = identity.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        let rotation = Tensor.empty(shape: [dim, dim], dtype: .bf16, device: device)
        rotation.copyIn(from: identityBits)
        let centroidValues = AURACodebook.centroids(dim: dim, bits: bits)
        let boundaryValues = AURACodebook.boundaries(dim: dim, bits: bits)
        let centroids = Tensor.empty(shape: [centroidValues.count], dtype: .bf16, device: device)
        centroids.copyIn(from: centroidValues.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) })
        let boundaries = Tensor.empty(shape: [boundaryValues.count], dtype: .bf16, device: device)
        boundaries.copyIn(from: boundaryValues.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) })

        let batchPacked = Tensor.empty(shape: [rows, packedWidth], dtype: .u32, device: device)
        let batchNorms = Tensor.empty(shape: [rows], dtype: .bf16, device: device)
        batchPacked.zero()
        batchNorms.zero()
        let singlePacked = Tensor.empty(shape: [rows, packedWidth], dtype: .u32, device: device)
        let singleNorms = Tensor.empty(shape: [rows], dtype: .bf16, device: device)
        singlePacked.zero()
        singleNorms.zero()

        let batchCommand = device.makeCommandBuffer()
        Ops.auraEncode(
            input: input, rotation: rotation, boundaries: boundaries, codebook: centroids,
            packedOut: batchPacked, normsOut: batchNorms,
            rows: rows, dim: dim, packedWidth: packedWidth, bits: bits, on: batchCommand)
        batchCommand.commit()
        batchCommand.waitUntilCompleted()

        let singleCommand = device.makeCommandBuffer()
        for row in 0 ..< rows {
            let inputView = Tensor(
                buffer: input.buffer, offset: input.offset + row * dim * 2,
                shape: [1, dim], dtype: .bf16)
            let packedView = Tensor(
                buffer: singlePacked.buffer,
                offset: singlePacked.offset + row * packedWidth * 4,
                shape: [1, packedWidth], dtype: .u32)
            let normView = Tensor(
                buffer: singleNorms.buffer, offset: singleNorms.offset + row * 2,
                shape: [1], dtype: .bf16)
            Ops.auraEncode(
                input: inputView, rotation: rotation, boundaries: boundaries, codebook: centroids,
                packedOut: packedView, normsOut: normView,
                rows: 1, dim: dim, packedWidth: packedWidth, bits: bits, on: singleCommand)
        }
        singleCommand.commit()
        singleCommand.waitUntilCompleted()

        #expect(batchPacked.toArray(as: UInt32.self) == singlePacked.toArray(as: UInt32.self))
        #expect(batchNorms.toArray(as: UInt16.self) == singleNorms.toArray(as: UInt16.self))
    }
}

// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

import Testing
import TestHelpers

@testable import ButterSwift

@Suite("Qwen3x compact draft rerank")
struct Qwen3xDraftRerankTests {
    private func bf16Bits(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(rounded >> 16)
    }

    @Test("bounded heap returns the exact strongest candidate set")
    func topCandidateSet() {
        var values: [Float] = []
        values.reserveCapacity(100)
        for i in 0 ..< 100 {
            let whole = Float((i * 37 + 11) % 101)
            let fraction = Float(i % 7) * 0.01
            values.append(whole + fraction)
        }
        values[17] = Float.nan
        values[23] = values[24]
        let limit = 90
        let count = Qwen3xMTPHead.draftRerankCandidateCount
        let actual = qwen3xTopDraftCandidateIDs(values, count: count, limit: limit)
        let expected = Array(
            (0 ..< limit).sorted { lhs, rhs in
                qwen3xDraftCandidateBetter(
                    lhsValue: values[lhs], lhsID: lhs,
                    rhsValue: values[rhs], rhsID: rhs)
            }.prefix(count))
        #expect(actual.count == count)
        #expect(Set(actual) == Set(expected))
    }

    @Test("equal values prefer the lower compact ID")
    func tieBreak() {
        let values = [Float](repeating: 4, count: 64)
        let count = Qwen3xMTPHead.draftRerankCandidateCount
        let actual = qwen3xTopDraftCandidateIDs(values, count: count, limit: values.count)
        #expect(Set(actual) == Set(0 ..< count))
    }

    @Test("tiled GPU selection matches the exact wide candidate set")
    func tiledCandidateSet() {
        let realCount = 98_330
        let paddedCount = 98_336
        let tileCount = 64
        let candidateCount = Qwen3xMTPHead.draftRerankCandidateCount
        var values = (0 ..< paddedCount).map {
            (Float(($0 * 73 + 29) % 65_521) - 32_760) / 1_024
        }
        values[137] = .nan
        values[41_223] = values[41_224]
        let bits = values.map(bf16Bits)
        let rounded = bits.map { Float(bitPattern: UInt32($0) << 16) }
        let expected = Array(
            (0 ..< realCount).sorted { lhs, rhs in
                qwen3xDraftCandidateBetter(
                    lhsValue: rounded[lhs], lhsID: lhs,
                    rhsValue: rounded[rhs], rhsID: rhs)
            }.prefix(candidateCount)
        ).map(UInt32.init)

        let logits = Tensor.empty(shape: [paddedCount], dtype: .bf16)
        logits.copyIn(from: bits)
        let scratch = Tensor.empty(
            shape: [tileCount * candidateCount], dtype: .u32)
        let indices = Tensor.empty(shape: [candidateCount], dtype: .u32)
        runAndWait { commandBuffer in
            Ops.topKIndicesTiled(
                logits: logits, candidateScratch: scratch,
                indicesOut: indices, count: realCount,
                tileCount: tileCount, topK: candidateCount,
                on: commandBuffer)
        }

        #expect(indices.toArray(as: UInt32.self) == expected)
    }

    @Test("compact control rows map to published token IDs")
    func compactMapping() {
        #expect(Qwen3xMTPHead.mapCompactDraftID(0) == 0)
        #expect(Qwen3xMTPHead.mapCompactDraftID(98_303) == 98_303)
        #expect(Qwen3xMTPHead.mapCompactDraftID(98_304) == 248_044)
        #expect(Qwen3xMTPHead.mapCompactDraftID(98_329) == 248_069)
    }

    @Test("wide row argmax matches the exact verifier geometry")
    func wideRowArgmax() {
        let rows = 4
        let columns = 248_320
        var values = (0 ..< rows * columns).map {
            Float(($0 * 73 + 29) % 1_009) / 1_024
        }
        var expected: [UInt32] = []
        for row in 0 ..< rows {
            let lower = 97 + row * 1_003
            let higher = lower + 65_536
            values[row * columns + lower] = 10
            values[row * columns + higher] = 10
            values[row * columns + lower + 1] = .nan
            expected.append(UInt32(lower))
        }

        let logits = Tensor.empty(shape: [rows, columns], dtype: .bf16)
        logits.copyIn(from: values.map(bf16Bits))
        let indices = Tensor.empty(shape: [rows], dtype: .u32)
        runAndWait { commandBuffer in
            Ops.argmaxRows(
                logits, rows: rows, columns: columns,
                into: indices, on: commandBuffer)
        }

        #expect(indices.toArray(as: UInt32.self) == expected)
    }

    @Test("compact mapping and indexed selection stay on GPU")
    func deviceResidentSelection() {
        let compact = Tensor.empty(shape: [4], dtype: .u32)
        compact.copyIn(from: [UInt32(0), 98_303, 98_304, 98_329])
        let full = Tensor.empty(shape: [4], dtype: .u32)
        let values = Tensor.empty(shape: [4], dtype: .bf16)
        values.copyIn(from: [Float(1), 7, .nan, 7].map(bf16Bits))
        let selected = Tensor.empty(shape: [1], dtype: .u32)

        runAndWait { commandBuffer in
            Ops.piecewiseOffsetIndices(
                compact, threshold: 98_304, offset: 149_740,
                into: full, on: commandBuffer)
            Ops.indexedArgmax(
                values, indices: full, into: selected, on: commandBuffer)
        }

        #expect(full.toArray(as: UInt32.self) == [0, 98_303, 248_044, 248_069])
        #expect(selected.toArray(as: UInt32.self) == [98_303])
    }

    @Test("gathered 4-bit rerank matches selected matrix rows")
    func gatheredRerankRows() {
        let outDim = 16
        let inDim = 512
        let groupSize = 64
        let groups = inDim / groupSize
        let selected: [UInt32] = [15, 0, 7, 3, 10, 4, 12, 1]

        var packed = [UInt32](repeating: 0, count: outDim * inDim / 8)
        for row in 0 ..< outDim {
            for col in 0 ..< inDim {
                let quantized = UInt32((row * 3 + col * 5) & 15)
                let index = row * inDim / 8 + col / 8
                packed[index] |= quantized << UInt32((col & 7) * 4)
            }
        }
        let scaleValues = [Float](repeating: 0.125, count: outDim * groups)
        let biasValues = [Float](repeating: -0.5, count: outDim * groups)
        let inputValues = (0 ..< inDim).map { Float(($0 * 11) % 29 - 14) / 16 }

        let weight = Tensor.empty(shape: [outDim, inDim / 8], dtype: .u32)
        weight.copyIn(from: packed)
        let scales = Tensor.empty(shape: [outDim, groups], dtype: .bf16)
        scales.copyIn(from: scaleValues.map(bf16Bits))
        let biases = Tensor.empty(shape: [outDim, groups], dtype: .bf16)
        biases.copyIn(from: biasValues.map(bf16Bits))
        let input = Tensor.empty(shape: [inDim], dtype: .bf16)
        input.copyIn(from: inputValues.map(bf16Bits))
        let rowIndices = Tensor.empty(shape: [selected.count], dtype: .u32)
        rowIndices.copyIn(from: selected)

        var output: Tensor!
        runAndWait { commandBuffer in
            output = Ops.dequantGemvInt4Gathered(
                weight: weight, scales: scales, biases: biases,
                rowIndices: rowIndices, input: input,
                groupSize: groupSize, on: commandBuffer)
        }

        let actual = output.toArray(as: UInt16.self).map {
            Float(bitPattern: UInt32($0) << 16)
        }
        for (candidate, selectedRow) in selected.enumerated() {
            let row = Int(selectedRow)
            var expected: Float = 0
            for col in 0 ..< inDim {
                let quantized = Float((row * 3 + col * 5) & 15)
                expected += (quantized * 0.125 - 0.5) * inputValues[col]
            }
            let tolerance = max(abs(expected) * 0.02, 0.5)
            #expect(
                abs(actual[candidate] - expected) <= tolerance,
                "candidate \(candidate), row \(row): got \(actual[candidate]), expected \(expected)")
        }
    }
}

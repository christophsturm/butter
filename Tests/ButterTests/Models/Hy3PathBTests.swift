// Copyright 2026 Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Pure Path B routing math — selection on sigmoid+bias, weights from
// unbiased, renorm, × router_scaling_factor. No Metal / no 92G weights.

import Foundation
import Testing

@testable import ButterSwift

@Suite("Hy3 Path B router math")
struct Hy3PathBTests {
    @Test("sigmoidBiasedTopK matches hand Path B (Hy3 shapes)")
    func pathBMatchesHandOracle() {
        let nExperts = 192
        let topK = 8
        let scale: Float = 2.826
        let logits: [Float] = (0 ..< nExperts).map { Float($0 % 31) * 0.1 - 1.5 }
        let bias: [Float] = (0 ..< nExperts).map { Float($0 % 7) * 0.05 - 0.15 }

        let router = MoERouter(
            nExperts: nExperts, topK: topK,
            gatingMode: .sigmoidBiasedTopK,
            normTopKProb: true,
            expertBias: bias,
            routedScalingFactor: scale)
        let routing = router.route(logits: logits)

        // Hand oracle
        var unbiased = [Float](repeating: 0, count: nExperts)
        var biased = [Float](repeating: 0, count: nExperts)
        for i in 0 ..< nExperts {
            let s = 1.0 / (1.0 + Foundation.exp(-logits[i]))
            unbiased[i] = s
            biased[i] = s + bias[i]
        }
        let idx = MoERouter.topKIndices(biased, k: topK)
        var weights = idx.map { unbiased[$0] }
        let sum = weights.reduce(0, +) + 1e-20
        weights = weights.map { $0 / sum * scale }

        #expect(routing.indices == idx)
        #expect(routing.weights.count == topK)
        for i in 0 ..< topK {
            #expect(abs(routing.weights[i] - weights[i]) < 1e-5)
        }
        let wSum = routing.weights.reduce(0, +)
        #expect(abs(wSum - scale) < 1e-4)
    }
}

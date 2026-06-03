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
import Foundation
import Metal
import TestHelpers
import Testing

@testable import FFAI

@Suite("Ops")
struct OpsTests {
    @Test("add f32 — c[i] = a[i] + b[i]")
    func addF32() {
        autoreleasepool {
            let a = Tensor.empty(shape: [5], dtype: .f32)
            let b = Tensor.empty(shape: [5], dtype: .f32)
            a.copyIn(from: [Float(1), 2, 3, 4, 5])
            b.copyIn(from: [Float(10), 20, 30, 40, 50])
            var out: Tensor!
            runAndWait { cb in out = Ops.add(a, b, on: cb) }
            #expect(out.toArray(as: Float.self) == [11, 22, 33, 44, 55])
        }
    }

    @Test("mul f32 — c[i] = a[i] * b[i]")
    func mulF32() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .f32)
            let b = Tensor.empty(shape: [4], dtype: .f32)
            a.copyIn(from: [Float(1), 2, 3, 4])
            b.copyIn(from: [Float(5), 6, 7, 8])
            var out: Tensor!
            runAndWait { cb in out = Ops.mul(a, b, on: cb) }
            #expect(out.toArray(as: Float.self) == [5, 12, 21, 32])
        }
    }

    @Test("im2colPatchInterleaved f32 — Gemma4 (py,px,c) unfold + affine + pad")
    func im2colPatchInterleavedF32() {
        autoreleasepool {
            let channels = 2, patch = 2, gridH = 1, gridW = 2
            let inH = gridH * patch, inW = gridW * patch
            let padded = 10  // real_dim = 8, two zero-pad columns
            let scale: Float = 2, bias: Float = -1
            var inData = [Float](repeating: 0, count: channels * inH * inW)
            for i in 0 ..< inData.count { inData[i] = Float(i) * 0.1 }
            let input = Tensor.empty(shape: [channels, inH, inW], dtype: .f32)
            input.copyIn(from: inData)
            // CPU oracle: channel-innermost columns, affine re-map, zero pad.
            let nPatches = gridH * gridW
            var expected = [Float](repeating: 0, count: nPatches * padded)
            for pr in 0 ..< gridH {
                for pc in 0 ..< gridW {
                    let pIdx = pr * gridW + pc
                    for py in 0 ..< patch {
                        for px in 0 ..< patch {
                            for c in 0 ..< channels {
                                let col = (py * patch + px) * channels + c
                                let v = inData[(c * inH + pr * patch + py) * inW + pc * patch + px]
                                expected[pIdx * padded + col] = v * scale + bias
                            }
                        }
                    }
                }
            }
            var out: Tensor!
            runAndWait { cb in
                out = Ops.im2colPatchInterleaved(
                    input, channels: channels, inH: inH, inW: inW, patch: patch,
                    patchDimPadded: padded, scale: scale, bias: bias, on: cb)
            }
            let got = out.toArray(as: Float.self)
            #expect(got.count == expected.count)
            for i in 0 ..< expected.count { #expect(abs(got[i] - expected[i]) < 1e-4) }
        }
    }

    @Test("patchUnfoldQwen f32 — temporal + merge-block raster, (t,c,py,px) cols")
    func patchUnfoldQwenF32() {
        autoreleasepool {
            let channels = 2, patch = 2, tp = 2, merge = 2, mergeBlocks = 1
            let side = mergeBlocks * merge
            let imgSide = side * patch
            let gridT = 1
            let isImage = true
            let padded = 18  // real_dim = tp*c*p*p = 16, two zero-pad columns
            let nFrames = isImage ? 1 : gridT * tp
            var frames = [Float](repeating: 0, count: nFrames * channels * imgSide * imgSide)
            for i in 0 ..< frames.count { frames[i] = Float(i) * 0.05 }
            let framesT = Tensor.empty(
                shape: [nFrames, channels, imgSide, imgSide], dtype: .f32)
            framesT.copyIn(from: frames)
            // CPU oracle: mirrors Qwen unfoldPatches exactly.
            let nPatches = gridT * side * side
            let frameStride = channels * imgSide * imgSide
            var expected = [Float](repeating: 0, count: nPatches * padded)
            var pIdx = 0
            for tg in 0 ..< gridT {
                for br in 0 ..< mergeBlocks {
                    for bc in 0 ..< mergeBlocks {
                        for ir in 0 ..< merge {
                            for ic in 0 ..< merge {
                                let pr = br * merge + ir
                                let pc = bc * merge + ic
                                var col = 0
                                for tw in 0 ..< tp {
                                    let fi = isImage ? 0 : tg * tp + tw
                                    for c in 0 ..< channels {
                                        for py in 0 ..< patch {
                                            for px in 0 ..< patch {
                                                let v = frames[
                                                    fi * frameStride
                                                        + (c * imgSide + pr * patch + py) * imgSide
                                                        + pc * patch + px]
                                                expected[pIdx * padded + col] = v
                                                col += 1
                                            }
                                        }
                                    }
                                }
                                pIdx += 1
                            }
                        }
                    }
                }
            }
            var out: Tensor!
            runAndWait { cb in
                out = Ops.patchUnfoldQwen(
                    frames: framesT, channels: channels, patch: patch, temporalPatch: tp,
                    merge: merge, mergeBlocks: mergeBlocks, imgSide: imgSide, gridT: gridT,
                    patchDimPadded: padded, isImage: isImage, on: cb)
            }
            let got = out.toArray(as: Float.self)
            #expect(got.count == expected.count)
            for i in 0 ..< expected.count { #expect(abs(got[i] - expected[i]) < 1e-4) }
        }
    }

    @Test("silu f32 — out[i] = x / (1 + exp(-x))")
    func siluF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(0), 1, -1, 2])
            var out: Tensor!
            runAndWait { cb in out = Ops.silu(x, on: cb) }
            let result = out.toArray(as: Float.self)
            // silu(0) = 0
            #expect(abs(result[0]) < 1e-5)
            // silu(1) ≈ 0.7311
            #expect(abs(result[1] - Float(1.0 / (1.0 + exp(-1.0)))) < 1e-3)
            // silu(-1) ≈ -0.2689
            #expect(abs(result[2] - Float(-1.0 / (1.0 + exp(1.0)))) < 1e-3)
            // silu(2) ≈ 1.7616
            #expect(abs(result[3] - Float(2.0 / (1.0 + exp(-2.0)))) < 1e-3)
        }
    }

    @Test("sigmoid f32 — out[i] = 1 / (1 + exp(-x))")
    func sigmoidF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(0), 1, -1, 2])
            var out: Tensor!
            runAndWait { cb in out = Ops.sigmoid(x, on: cb) }
            let r = out.toArray(as: Float.self)
            // sigmoid(0) = 0.5
            #expect(abs(r[0] - 0.5) < 1e-5)
            // sigmoid(1) ≈ 0.7311
            #expect(abs(r[1] - Float(1.0 / (1.0 + exp(-1.0)))) < 1e-3)
            // sigmoid(-1) ≈ 0.2689
            #expect(abs(r[2] - Float(1.0 / (1.0 + exp(1.0)))) < 1e-3)
            // sigmoid(2) ≈ 0.8808
            #expect(abs(r[3] - Float(1.0 / (1.0 + exp(-2.0)))) < 1e-3)
        }
    }

    @Test("ropePartial f32 — rotates only the first rotaryDim of each head")
    func ropePartialRotatesSubset() {
        autoreleasepool {
            // headDim=4, rotaryDim=2: a single head, rotate dims [0,1),
            // pass through dims [2,4). Position 1, theta_base=10000.
            let qk = Tensor.empty(shape: [4], dtype: .f32)
            qk.copyIn(from: [Float(1), 0, 7, 9])
            runAndWait { cb in
                Ops.ropePartial(
                    qk, position: 1, headDim: 4, rotaryDim: 2,
                    thetaBase: 10000, on: cb)
            }
            let r = qk.toArray(as: Float.self)
            // rotaryDim=2 → one rotate-half pair (0, 1). inv_freq=1,
            // theta=1, cos≈0.5403, sin≈0.8415.
            //   r[0] = x[0]*cos - x[1]*sin = 1*0.5403 - 0*0.8415 = 0.5403
            //   r[1] = x[0]*sin + x[1]*cos = 1*0.8415 + 0*0.5403 = 0.8415
            #expect(abs(r[0] - 0.5403) < 1e-3)
            #expect(abs(r[1] - 0.8415) < 1e-3)
            // Dims [2,4) are outside rotaryDim → untouched pass-through.
            #expect(abs(r[2] - 7) < 1e-5)
            #expect(abs(r[3] - 9) < 1e-5)
        }
    }

    @Test("ropePartialTwo f32 — same result as two sequential ropePartial calls")
    func ropePartialTwoMatchesSequential() {
        autoreleasepool {
            let headDim = 4
            let rotaryDim = 2
            let q0 = Tensor.empty(shape: [headDim], dtype: .f32)
            q0.copyIn(from: [Float(1), 0, 7, 9])
            let k0 = Tensor.empty(shape: [headDim], dtype: .f32)
            k0.copyIn(from: [Float(0), 1, -3, 4])
            let q1 = Tensor.empty(shape: [headDim], dtype: .f32)
            q1.copyIn(from: [Float(1), 0, 7, 9])
            let k1 = Tensor.empty(shape: [headDim], dtype: .f32)
            k1.copyIn(from: [Float(0), 1, -3, 4])
            // Reference: two independent ropePartial calls.
            runAndWait { cb in
                Ops.ropePartial(
                    q0, position: 3, headDim: headDim, rotaryDim: rotaryDim,
                    thetaBase: 10000, on: cb)
                Ops.ropePartial(
                    k0, position: 3, headDim: headDim, rotaryDim: rotaryDim,
                    thetaBase: 10000, on: cb)
            }
            // Batched: one-encoder, two dispatches.
            runAndWait { cb in
                Ops.ropePartialTwo(
                    q1, k1, position: 3, headDim: headDim, rotaryDim: rotaryDim,
                    thetaBase: 10000, on: cb)
            }
            let r0q = q0.toArray(as: Float.self)
            let r0k = k0.toArray(as: Float.self)
            let r1q = q1.toArray(as: Float.self)
            let r1k = k1.toArray(as: Float.self)
            for i in 0 ..< headDim {
                #expect(abs(r0q[i] - r1q[i]) < 1e-5, "q[\(i)]: \(r0q[i]) vs \(r1q[i])")
                #expect(abs(r0k[i] - r1k[i]) < 1e-5, "k[\(i)]: \(r0k[i]) vs \(r1k[i])")
            }
        }
    }

    @Test("ropePartialMany f32 — T=3 matches three sequential ropePartial calls")
    func ropePartialManyMatchesSequential() {
        autoreleasepool {
            let t = 3
            let nHeads = 2
            let headDim = 4
            let rotaryDim = 2
            let rowStride = nHeads * headDim
            let total = t * rowStride
            let data: [Float] = (0 ..< total).map { Float($0 + 1) * 0.1 }
            // Reference: T sequential single-token rotations against
            // T separate buffers (positions 0, 1, 2).
            var refRows: [[Float]] = []
            for tt in 0 ..< t {
                let slice = Tensor.empty(shape: [nHeads, headDim], dtype: .f32)
                slice.copyIn(from: Array(data[tt * rowStride ..< (tt + 1) * rowStride]))
                runAndWait { cb in
                    Ops.ropePartial(
                        slice, position: tt, headDim: headDim, rotaryDim: rotaryDim,
                        thetaBase: 10000, on: cb)
                }
                refRows.append(slice.toArray(as: Float.self))
            }
            // Batched: single dispatch with positions [0, 1, 2].
            let qk = Tensor.empty(shape: [t, nHeads, headDim], dtype: .f32)
            qk.copyIn(from: data)
            let positions = Tensor.empty(shape: [t], dtype: .u32)
            positions.copyIn(from: (0 ..< t).map { UInt32($0) })
            runAndWait { cb in
                Ops.ropePartialMany(
                    qk, positions: positions,
                    t: t, nHeads: nHeads,
                    headDim: headDim, rotaryDim: rotaryDim,
                    rowStride: rowStride,
                    thetaBase: 10000, on: cb)
            }
            let got = qk.toArray(as: Float.self)
            for tt in 0 ..< t {
                let row = refRows[tt]
                for i in 0 ..< rowStride {
                    #expect(
                        abs(got[tt * rowStride + i] - row[i]) < 1e-5,
                        "t=\(tt) i=\(i): \(got[tt * rowStride + i]) vs \(row[i])")
                }
            }
        }
    }

    @Test("ropePartial f32 — rotaryDim == headDim matches full rope")
    func ropePartialFullEqualsRope() {
        autoreleasepool {
            let a = Tensor.empty(shape: [4], dtype: .f32)
            a.copyIn(from: [Float(1), 0, 0, 1])
            let b = Tensor.empty(shape: [4], dtype: .f32)
            b.copyIn(from: [Float(1), 0, 0, 1])
            var full: Tensor!
            runAndWait { cb in
                full = Ops.rope(
                    a, position: 1, headDim: 4,
                    thetaBase: 10000, on: cb)
            }
            runAndWait { cb in
                Ops.ropePartial(
                    b, position: 1, headDim: 4, rotaryDim: 4,
                    thetaBase: 10000, on: cb)
            }
            let rf = full.toArray(as: Float.self)
            let rp = b.toArray(as: Float.self)
            for i in 0 ..< 4 {
                #expect(abs(rf[i] - rp[i]) < 1e-4, "i=\(i): \(rf[i]) vs \(rp[i])")
            }
        }
    }

    @Test("relu f32 — out[i] = max(x[i], 0)")
    func reluF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [5], dtype: .f32)
            x.copyIn(from: [Float(0), 1, -1, 2, -3.5])
            var out: Tensor!
            runAndWait { cb in out = Ops.relu(x, on: cb) }
            let result = out.toArray(as: Float.self)
            #expect(result == [0, 1, 0, 2, 0])
        }
    }

    @Test("squared-relu via relu + mul — NemotronH MLP activation")
    func squaredReluF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(-2), 0, 1.5, 3])
            var out: Tensor!
            runAndWait { cb in
                let r = Ops.relu(x, on: cb)
                out = Ops.mul(r, r, on: cb)
            }
            let result = out.toArray(as: Float.self)
            // relu(x)^2: negatives clamp to 0, positives square.
            #expect(result == [0, 0, 2.25, 9])
        }
    }

    @Test("gelu f32 — out[i] = 0.5 * x * (1 + tanh(...))")
    func geluF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [4], dtype: .f32)
            x.copyIn(from: [Float(0), 1, -1, 2])
            var out: Tensor!
            runAndWait { cb in out = Ops.gelu(x, on: cb) }
            let result = out.toArray(as: Float.self)
            // tanh-approx GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
            func ref(_ v: Float) -> Float {
                let c: Float = Float((2.0 / Double.pi).squareRoot())
                let inner = c * (v + 0.044715 * v * v * v)
                return 0.5 * v * (1 + tanh(inner))
            }
            #expect(abs(result[0] - ref(0)) < 1e-3)
            #expect(abs(result[1] - ref(1)) < 1e-3)
            #expect(abs(result[2] - ref(-1)) < 1e-3)
            #expect(abs(result[3] - ref(2)) < 1e-3)
        }
    }

    @Test("gelu bf16 — extreme inputs stay finite (Gemma 3 regression)")
    func geluBf16ExtremeInputs() {
        // Gemma 3 1B layer-1 gate values span [-10.25, 10.81] in bf16.
        // The tanh argument hits k*(x + 0.044715*x^3) ≈ ±54 there,
        // beyond Metal's native bf16 tanh range and beyond what an
        // (exp(2x)-1)/(exp(2x)+1) evaluation can survive in bf16's
        // 8-bit exponent. Test that gelu(bf16) over the failing range
        // produces only finite values — no NaN, no inf.
        autoreleasepool {
            let xs: [Float] = stride(from: Float(-15.0), through: 15.0, by: 0.25).map { $0 }
            let n = xs.count
            let x = Tensor.empty(shape: [n], dtype: .bf16)
            // Pack fp32 → bf16 via FFAI's existing test helper.
            let xPtr = x.buffer.contents().bindMemory(to: UInt16.self, capacity: n)
            for (i, v) in xs.enumerated() {
                xPtr[i] = floatToBf16BitsForTest(v)
            }
            var out: Tensor!
            runAndWait { cb in out = Ops.gelu(x, on: cb) }
            let outPtr = out.buffer.contents().bindMemory(to: UInt16.self, capacity: n)
            for i in 0 ..< n {
                let v = bf16BitsToFloatForTest(outPtr[i])
                #expect(
                    v.isFinite,
                    "gelu(\(xs[i])) in bf16 → \(v); must be finite")
            }
        }
    }

    @Test("gather f32 — picks the right rows")
    func gatherF32() {
        autoreleasepool {
            // table[3, 2] = [[10,11], [20,21], [30,31]]
            let table = Tensor.empty(shape: [3, 2], dtype: .f32)
            table.copyIn(from: [Float(10), 11, 20, 21, 30, 31])
            let ids = Tensor.empty(shape: [2], dtype: .u32)
            ids.copyIn(from: [UInt32(2), 0])
            var out: Tensor!
            runAndWait { cb in out = Ops.gather(table: table, tokenIds: ids, on: cb) }
            #expect(out.shape == [2, 2])
            #expect(out.toArray(as: Float.self) == [30, 31, 10, 11])
        }
    }

    @Test("gatherTwo f32 — two ids streams against one table on one encoder")
    func gatherTwoF32() {
        autoreleasepool {
            let table = Tensor.empty(shape: [3, 2], dtype: .f32)
            table.copyIn(from: [Float(10), 11, 20, 21, 30, 31])
            let ids1 = Tensor.empty(shape: [2], dtype: .u32)
            ids1.copyIn(from: [UInt32(2), 0])
            let ids2 = Tensor.empty(shape: [3], dtype: .u32)
            ids2.copyIn(from: [UInt32(1), 1, 2])
            let out1 = Tensor.empty(shape: [2, 2], dtype: .f32)
            let out2 = Tensor.empty(shape: [3, 2], dtype: .f32)
            runAndWait { cb in
                Ops.gatherTwo(
                    table: table,
                    ids1: ids1, into: out1,
                    ids2: ids2, into: out2,
                    on: cb)
            }
            #expect(out1.toArray(as: Float.self) == [30, 31, 10, 11])
            #expect(out2.toArray(as: Float.self) == [20, 21, 20, 21, 30, 31])
        }
    }

    @Test("gemv f32 — out[i] = sum_j W[i,j] * x[j]")
    func gemvF32() {
        autoreleasepool {
            // W [3, 2] = [[1,2], [3,4], [5,6]]
            let w = Tensor.empty(shape: [3, 2], dtype: .f32)
            w.copyIn(from: [Float(1), 2, 3, 4, 5, 6])
            let x = Tensor.empty(shape: [2], dtype: .f32)
            x.copyIn(from: [Float(7), 8])
            // expected: [1*7+2*8, 3*7+4*8, 5*7+6*8] = [23, 53, 83]
            var out: Tensor!
            runAndWait { cb in out = Ops.gemv(weight: w, input: x, on: cb) }
            #expect(out.toArray(as: Float.self) == [23, 53, 83])
        }
    }

    @Test("auraRotatePerHead f32 — permutation matrix applied per head")
    func auraRotatePerHeadPermutation() {
        autoreleasepool {
            // headDim = 4, nHeads = 3. Rotation is the cyclic permutation
            // P[i, j] = 1 iff j == (i + 1) % 4, else 0; row-major.
            // (P · v)[i] = v[(i+1) % 4]  — shifts each head's slice left
            // by one, with wraparound. Use distinct values per head so a
            // bug that crosses head boundaries surfaces immediately.
            let headDim = 4
            let nHeads = 3
            var rot = [Float](repeating: 0, count: headDim * headDim)
            for i in 0 ..< headDim {
                rot[i * headDim + ((i + 1) % headDim)] = 1
            }
            let rotation = Tensor.empty(shape: [headDim, headDim], dtype: .f32)
            rotation.copyIn(from: rot)

            // Per-head input: [10,11,12,13, 20,21,22,23, 30,31,32,33].
            var inVals = [Float]()
            for h in 0 ..< nHeads {
                for i in 0 ..< headDim {
                    inVals.append(Float((h + 1) * 10 + i))
                }
            }
            let x = Tensor.empty(shape: [nHeads * headDim], dtype: .f32)
            x.copyIn(from: inVals)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.auraRotatePerHead(
                    x, rotation: rotation,
                    nHeads: nHeads, headDim: headDim, on: cb)
            }
            let got = out.toArray(as: Float.self)

            // Expected: each head's slice cyclically shifted left by one.
            //   head 0: [10,11,12,13] → [11,12,13,10]
            //   head 1: [20,21,22,23] → [21,22,23,20]
            //   head 2: [30,31,32,33] → [31,32,33,30]
            var expected = [Float]()
            for h in 0 ..< nHeads {
                for i in 0 ..< headDim {
                    expected.append(Float((h + 1) * 10 + ((i + 1) % headDim)))
                }
            }
            #expect(got == expected)
        }
    }

    @Test("auraRotatePerHead f32 — identity rotation is a no-op")
    func auraRotatePerHeadIdentity() {
        autoreleasepool {
            // Identity rotation must round-trip the input exactly; this
            // pins the "no rotation" baseline path the AURA cache used
            // pre-SRHT and matches what the AURARotation.identityMatrix
            // helper emits.
            let headDim = 8
            let nHeads = 2
            var rot = [Float](repeating: 0, count: headDim * headDim)
            for i in 0 ..< headDim { rot[i * headDim + i] = 1 }
            let rotation = Tensor.empty(shape: [headDim, headDim], dtype: .f32)
            rotation.copyIn(from: rot)

            let inVals: [Float] = (0 ..< (nHeads * headDim)).map { Float($0) * 0.5 - 1.25 }
            let x = Tensor.empty(shape: [nHeads * headDim], dtype: .f32)
            x.copyIn(from: inVals)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.auraRotatePerHead(
                    x, rotation: rotation,
                    nHeads: nHeads, headDim: headDim, on: cb)
            }
            let got = out.toArray(as: Float.self)
            for i in 0 ..< inVals.count {
                #expect(
                    abs(got[i] - inVals[i]) < 1e-5,
                    "i=\(i) got \(got[i]) expected \(inVals[i])")
            }
        }
    }

    @Test("rmsNorm f32 — y = x / rms(x) * weight")
    func rmsNormF32() {
        autoreleasepool {
            // Use the smallest valid size for this kernel: n must be a
            // multiple of 128 (32-lane simdgroup × 4 elements/thread).
            // See Ops.rmsNorm preconditions for the full constraint set.
            let n = 128
            let xs: [Float] = (0 ..< n).map { Float($0 + 1) }  // [1, 2, …, 128]
            let ws: [Float] = Array(repeating: Float(1), count: n)

            let x = Tensor.empty(shape: [n], dtype: .f32)
            x.copyIn(from: xs)
            let weight = Tensor.empty(shape: [n], dtype: .f32)
            weight.copyIn(from: ws)

            var out: Tensor!
            runAndWait { cb in out = Ops.rmsNorm(x, weight: weight, eps: 1e-6, on: cb) }
            let result = out.toArray(as: Float.self)

            // CPU reference: rms = sqrt(mean(x^2)); y = x / rms * weight.
            let ssq = xs.reduce(Float(0)) { $0 + $1 * $1 }
            let expectedRms = (ssq / Float(n)).squareRoot()
            for i in 0 ..< n {
                let expected = xs[i] / expectedRms
                #expect(
                    abs(result[i] - expected) < 1e-2,
                    "i=\(i) got \(result[i]) expected \(expected)")
            }
        }
    }

    @Test("rmsNormRowsTwo f32 — matches two sequential rmsNormRows calls")
    func rmsNormRowsTwoMatchesSequential() {
        autoreleasepool {
            let n1 = 128
            let n2 = 256
            let rows = 1
            let xs1: [Float] = (0 ..< n1 * rows).map { Float($0 + 1) }
            let ws1: [Float] = (0 ..< n1).map { 1.0 + Float($0 % 7) * 0.05 }
            let xs2: [Float] = (0 ..< n2 * rows).map { Float($0 + 1) * 0.3 }
            let ws2: [Float] = (0 ..< n2).map { 1.0 + Float($0 % 11) * 0.03 }
            // Reference: two sequential single-row rmsNormRows calls.
            let x1Ref = Tensor.empty(shape: [rows, n1], dtype: .f32)
            x1Ref.copyIn(from: xs1)
            let w1Ref = Tensor.empty(shape: [n1], dtype: .f32)
            w1Ref.copyIn(from: ws1)
            let x2Ref = Tensor.empty(shape: [rows, n2], dtype: .f32)
            x2Ref.copyIn(from: xs2)
            let w2Ref = Tensor.empty(shape: [n2], dtype: .f32)
            w2Ref.copyIn(from: ws2)
            var ref1: Tensor!
            var ref2: Tensor!
            runAndWait { cb in
                ref1 = Ops.rmsNormRows(
                    x1Ref, weight: w1Ref, eps: 1e-6,
                    nRows: rows, rowSize: n1, on: cb)
                ref2 = Ops.rmsNormRows(
                    x2Ref, weight: w2Ref, eps: 1e-6,
                    nRows: rows, rowSize: n2, on: cb)
            }
            // Batched: same eps so the wrapper shares one alloc.
            let x1 = Tensor.empty(shape: [rows, n1], dtype: .f32)
            x1.copyIn(from: xs1)
            let w1 = Tensor.empty(shape: [n1], dtype: .f32)
            w1.copyIn(from: ws1)
            let x2 = Tensor.empty(shape: [rows, n2], dtype: .f32)
            x2.copyIn(from: xs2)
            let w2 = Tensor.empty(shape: [n2], dtype: .f32)
            w2.copyIn(from: ws2)
            let out1 = Tensor.empty(shape: [rows, n1], dtype: .f32)
            let out2 = Tensor.empty(shape: [rows, n2], dtype: .f32)
            runAndWait { cb in
                Ops.rmsNormRowsTwo(
                    x1, weight: w1, eps1: 1e-6,
                    nRows1: rows, rowSize1: n1, into: out1,
                    x2, weight: w2, eps2: 1e-6,
                    nRows2: rows, rowSize2: n2, into: out2,
                    on: cb)
            }
            let r1Ref = ref1.toArray(as: Float.self)
            let r2Ref = ref2.toArray(as: Float.self)
            let r1 = out1.toArray(as: Float.self)
            let r2 = out2.toArray(as: Float.self)
            for i in 0 ..< n1 {
                #expect(abs(r1[i] - r1Ref[i]) < 1e-4, "norm1[\(i)]: \(r1[i]) vs \(r1Ref[i])")
            }
            for i in 0 ..< n2 {
                #expect(abs(r2[i] - r2Ref[i]) < 1e-4, "norm2[\(i)]: \(r2[i]) vs \(r2Ref[i])")
            }
        }
    }

    @Test("rmsNorm f32 — wide row (n=5376) routes to mt_rms_norm_wide")
    func rmsNormWideF32() {
        autoreleasepool {
            // n = 5376 (Gemma 4 31B hidden) is past the 4096 cap of the
            // 4-elements-per-thread kernel, so Ops.rmsNorm routes to the
            // strided mt_rms_norm_wide kernel. eps eps=1e-6 like Gemma 4.
            let n = 5376
            let eps: Float = 1e-6
            let xs: [Float] = (0 ..< n).map { Float(($0 % 37) - 18) * 0.21 }
            let ws: [Float] = (0 ..< n).map { 1.0 + Float($0 % 11) * 0.03 }

            let x = Tensor.empty(shape: [n], dtype: .f32)
            x.copyIn(from: xs)
            let weight = Tensor.empty(shape: [n], dtype: .f32)
            weight.copyIn(from: ws)

            var out: Tensor!
            runAndWait { cb in out = Ops.rmsNorm(x, weight: weight, eps: eps, on: cb) }
            let result = out.toArray(as: Float.self)

            // CPU reference: rms = sqrt(mean(x^2) + eps); y = x/rms*weight.
            let ssq = xs.reduce(Float(0)) { $0 + $1 * $1 }
            let expectedRms = (ssq / Float(n) + eps).squareRoot()
            for i in 0 ..< n {
                let expected = xs[i] / expectedRms * ws[i]
                #expect(
                    abs(result[i] - expected) < 1e-3,
                    "i=\(i) got \(result[i]) expected \(expected)")
            }
        }
    }

    @Test("rope f32 at position 0 is identity (cos=1, sin=0)")
    func ropePos0Identity() {
        autoreleasepool {
            let qk = Tensor.empty(shape: [1, 4], dtype: .f32)
            qk.copyIn(from: [Float(1), 2, 3, 4])
            var out: Tensor!
            runAndWait { cb in
                out = Ops.rope(qk, position: 0, headDim: 4, thetaBase: 10000, on: cb)
            }
            let r = out.toArray(as: Float.self)
            // theta = 0 → cos = 1, sin = 0 → identity
            for i in 0 ..< 4 {
                #expect(abs(r[i] - Float(i + 1)) < 1e-4, "i=\(i) got \(r[i])")
            }
        }
    }

    @Test("rope f32 at position 1, head_dim=4, theta_base=10000")
    func ropePos1() {
        autoreleasepool {
            let qk = Tensor.empty(shape: [1, 4], dtype: .f32)
            qk.copyIn(from: [Float(1), 0, 0, 1])
            var out: Tensor!
            runAndWait { cb in
                out = Ops.rope(qk, position: 1, headDim: 4, thetaBase: 10000, on: cb)
            }
            // i_pair=0: inv_freq = 1, theta = 1, cos≈0.5403, sin≈0.8415
            //   x[0] * cos - x[2] * sin = 1*0.5403 - 0*0.8415 = 0.5403
            //   x[0] * sin + x[2] * cos = 1*0.8415 + 0*0.5403 = 0.8415
            // i_pair=1: inv_freq = 1/sqrt(10000) = 0.01, theta = 0.01, cos≈1, sin≈0.01
            //   x[1]*cos - x[3]*sin = 0*1 - 1*0.01 = -0.01
            //   x[1]*sin + x[3]*cos = 0*0.01 + 1*1 = 1
            let r = out.toArray(as: Float.self)
            #expect(abs(r[0] - 0.5403) < 1e-3)  // index 0
            #expect(abs(r[1] - (-0.01)) < 5e-3)  // index 1 (looser tol for f32 trig)
            #expect(abs(r[2] - 0.8415) < 1e-3)  // index 0 + half_dim
            #expect(abs(r[3] - 1.0) < 1e-3)  // index 1 + half_dim
        }
    }

    @Test("sdpaDecode f32 — single position attends to itself")
    func sdpaSinglePosition() {
        autoreleasepool {
            // Kernel invariant: head_dim must be 128 (32 simdgroups × 32 lanes ×
            // 4 elements/lane). Below 128 the wrapper preconditions catch it;
            // before the preconditions existed, the test ran with head_dim=4
            // and pinned the GPU.
            let D = 128
            let kvStride = 4  // pre-allocated capacity
            let nKV = 1  // only the first position is filled
            let nQHeads = 1
            let nKVHeads = 1

            let q = Tensor.empty(shape: [nQHeads, D], dtype: .f32)
            let k = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)
            let v = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)

            // Q is the unit vector e_0; K[0] is the same so dot(Q, K[0]) = 1.
            var qData = [Float](repeating: 0, count: nQHeads * D)
            qData[0] = 1
            q.copyIn(from: qData)

            var kData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            kData[0] = 1  // K[head=0, pos=0, d=0]
            k.copyIn(from: kData)

            // V[0] is an arbitrary recognizable vector; positions [1..3]
            // are zero so even if the kernel read past `n_kv` we'd notice.
            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            for d in 0 ..< D { vData[d] = Float(d + 1) }  // [1, 2, …, 128]
            v.copyIn(from: vData)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.sdpaDecode(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads,
                    headDim: D,
                    nKV: nKV, kvStride: kvStride,
                    scale: 1.0, on: cb)
            }
            // n_kv = 1 → softmax([single_score]) = 1 → output == V[0].
            let r = out.toArray(as: Float.self)
            for d in 0 ..< D {
                #expect(
                    abs(r[d] - Float(d + 1)) < 1e-4,
                    "out[\(d)] = \(r[d]), expected \(d + 1)")
            }
        }
    }

    @Test("sdpaDecodeSink f32 — per-head learned sink folds into the softmax denom (d64 GQA)")
    func sdpaDecodeSinkF32() {
        autoreleasepool {
            // GPT-OSS decode shape: d64, GQA (4 q-heads → 2 kv-heads).
            let D = 64
            let nQHeads = 4
            let nKVHeads = 2
            let headsPerGroup = nQHeads / nKVHeads
            let nKV = 5
            let kvStride = 8  // capacity > nKV so stale rows must be ignored
            let scale = 1.0 / Float(Double(D).squareRoot())

            func ramp(_ n: Int, _ seed: Int, _ amp: Float) -> [Float] {
                (0 ..< n).map { (Float(($0 + seed) % 11) - 5) * amp / 11 }
            }
            let qData = ramp(nQHeads * D, 3, 1.2)
            let kData = ramp(nKVHeads * kvStride * D, 7, 0.9)
            let vData = ramp(nKVHeads * kvStride * D, 5, 0.7)
            let sinkData: [Float] = [0.4, -0.7, 1.1, 0.0]  // per q-head logits

            let q = Tensor.empty(shape: [nQHeads, D], dtype: .f32); q.copyIn(from: qData)
            let k = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32); k.copyIn(from: kData)
            let v = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32); v.copyIn(from: vData)
            let sink = Tensor.empty(shape: [nQHeads], dtype: .f32); sink.copyIn(from: sinkData)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.sdpaDecodeSink(
                    q: q, k: k, v: v, sink: sink,
                    nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: D,
                    nKV: nKV, kvStride: kvStride, scale: scale, on: cb)
            }
            let got = out.toArray(as: Float.self)

            // CPU oracle — mirrors sdpa_decode_sink_buf.rs: the sink is a
            // virtual key (score sink[qh], value 0) included in the global
            // max + denominator. Dense path (sinkEnd = windowStart = 0).
            var maxDiff: Float = 0
            for qh in 0 ..< nQHeads {
                let kvh = qh / headsPerGroup
                var scores = [Float](repeating: 0, count: nKV)
                for t in 0 ..< nKV {
                    var dot: Float = 0
                    for d in 0 ..< D {
                        dot += qData[qh * D + d] * kData[(kvh * kvStride + t) * D + d]
                    }
                    scores[t] = dot * scale
                }
                let gMax = max(scores.max() ?? 0, sinkData[qh])
                var denom = Foundation.exp(sinkData[qh] - gMax)
                for t in 0 ..< nKV { denom += Foundation.exp(scores[t] - gMax) }
                for d in 0 ..< D {
                    var acc: Float = 0
                    for t in 0 ..< nKV {
                        acc += Foundation.exp(scores[t] - gMax) * vData[(kvh * kvStride + t) * D + d]
                    }
                    let expected = acc / denom
                    maxDiff = max(maxDiff, abs(got[qh * D + d] - expected))
                }
            }
            #expect(maxDiff < 1e-4, "sdpaDecodeSink diverges from oracle (max \(maxDiff))")
        }
    }

    @Test("sdpaDecode f32 — sinkEnd + windowStart skip the masked range")
    func sdpaSlidingWindowAndSinks() {
        autoreleasepool {
            // head_dim=128 is the only variant carrying the sink/window
            // constexprs. Four KV positions; the kernel attends
            // [0, sinkEnd) ∪ [windowStart, nKV) and skips the gap.
            let D = 128
            let kvStride = 4
            let nKV = 4
            let nQHeads = 1
            let nKVHeads = 1

            let q = Tensor.empty(shape: [nQHeads, D], dtype: .f32)
            let k = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)
            let v = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)

            // Q = e_0. Every K row is also e_0, so all four positions
            // share the same score → softmax weights them equally over
            // whatever subset the kernel actually attends.
            var qData = [Float](repeating: 0, count: nQHeads * D)
            qData[0] = 1
            q.copyIn(from: qData)

            var kData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            for pos in 0 ..< kvStride { kData[pos * D + 0] = 1 }
            k.copyIn(from: kData)

            // V[pos] = constant vector of value `pos`. The attention
            // output's first element is the mean of the attended `pos`
            // values, which makes the attended set directly checkable.
            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            for pos in 0 ..< kvStride {
                for d in 0 ..< D { vData[pos * D + d] = Float(pos) }
            }
            v.copyIn(from: vData)

            // windowStart=2, sinkEnd=0 → attends positions {2, 3} only.
            // mean(2, 3) = 2.5.
            var windowed: Tensor!
            runAndWait { cb in
                windowed = Ops.sdpaDecode(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads,
                    headDim: D, nKV: nKV, kvStride: kvStride,
                    scale: 1.0, on: cb,
                    sinkEnd: 0, windowStart: 2)
            }
            let wr = windowed.toArray(as: Float.self)
            #expect(
                abs(wr[0] - 2.5) < 1e-4,
                "windowStart=2 should attend {2,3}: got \(wr[0]), expected 2.5")

            // sinkEnd=1, windowStart=3 → attends {0} ∪ {3}.
            // mean(0, 3) = 1.5.
            var sinked: Tensor!
            runAndWait { cb in
                sinked = Ops.sdpaDecode(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads,
                    headDim: D, nKV: nKV, kvStride: kvStride,
                    scale: 1.0, on: cb,
                    sinkEnd: 1, windowStart: 3)
            }
            let sr = sinked.toArray(as: Float.self)
            #expect(
                abs(sr[0] - 1.5) < 1e-4,
                "sinkEnd=1, windowStart=3 should attend {0,3}: got \(sr[0]), expected 1.5")

            // sinkEnd=0, windowStart=0 → dense full attention {0,1,2,3}.
            // mean(0,1,2,3) = 1.5 — same mean as above by coincidence,
            // so cross-check that an explicit dense call equals the
            // default-argument call.
            var dense: Tensor!
            runAndWait { cb in
                dense = Ops.sdpaDecode(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads,
                    headDim: D, nKV: nKV, kvStride: kvStride,
                    scale: 1.0, on: cb,
                    sinkEnd: 0, windowStart: 0)
            }
            let dr = dense.toArray(as: Float.self)
            #expect(
                abs(dr[0] - 1.5) < 1e-4,
                "dense attention over {0,1,2,3}: got \(dr[0]), expected 1.5")
        }
    }

    // MARK: - softmax_categorical_sample

    /// Helper: run the GPU sample once with a given uniform draw, return token id.
    private func runGPUSample(logits: Tensor, temperature: Float, uniform: Float) -> Int {
        let tBuf = Tensor.empty(shape: [1], dtype: .f32)
        tBuf.copyIn(from: [temperature])
        let uBuf = Tensor.empty(shape: [1], dtype: .f32)
        uBuf.copyIn(from: [uniform])
        let out = Tensor.empty(shape: [1], dtype: .u32)
        runAndWait { cb in
            Ops.softmaxCategoricalSample(
                logits, into: out, temperature: tBuf,
                uniform: uBuf, on: cb)
        }
        return Int(out.toArray(as: UInt32.self)[0])
    }

    @Test("GPU sample: peaked distribution always picks the peak", .disabled("bisect"))
    func gpuSamplePeaked() {
        autoreleasepool {
            let l = Tensor.empty(shape: [5], dtype: .f32)
            l.copyIn(from: [Float(0), 0, 10, 0, 0])
            for u in [Float(0.001), 0.5, 0.999] {
                #expect(runGPUSample(logits: l, temperature: 1.0, uniform: u) == 2)
            }
        }
    }

    @Test("GPU sample: uniform logits + uniform=0.0 picks index 0", .disabled("bisect"))
    func gpuSampleUniformLow() {
        autoreleasepool {
            let l = Tensor.empty(shape: [4], dtype: .f32)
            l.copyIn(from: [Float(1), 1, 1, 1])
            #expect(runGPUSample(logits: l, temperature: 1.0, uniform: 0.0) == 0)
        }
    }

    @Test("GPU sample: uniform logits + uniform near 1 picks last index", .disabled("bisect"))
    func gpuSampleUniformHigh() {
        autoreleasepool {
            let l = Tensor.empty(shape: [4], dtype: .f32)
            l.copyIn(from: [Float(1), 1, 1, 1])
            #expect(runGPUSample(logits: l, temperature: 1.0, uniform: 0.99) == 3)
        }
    }

    @Test("GPU sample: matches CPU CDF walk over a 32-vocab sweep", .disabled("bisect"))
    func gpuSampleMatchesCPU() {
        autoreleasepool {
            let n = 32
            var vals = [Float](repeating: 0, count: n)
            for i in 0 ..< n { vals[i] = Float(i) / 5.0 }
            let l = Tensor.empty(shape: [n], dtype: .f32)
            l.copyIn(from: vals)

            let T: Float = 1.5
            let invT = 1.0 / T
            let scaled = vals.map { $0 * invT }
            let maxL = scaled.max() ?? 0
            let expSums = scaled.map { Foundation.exp(Double($0 - maxL)) }
            let total = expSums.reduce(0, +)
            func cpuExpected(uniform: Float) -> Int {
                let target = Double(uniform) * total
                var cum = 0.0
                for i in 0 ..< n {
                    cum += expSums[i]
                    if cum >= target { return i }
                }
                return n - 1
            }
            for u in stride(from: Float(0.05), to: 1.0, by: 0.1) {
                let gpu = runGPUSample(logits: l, temperature: T, uniform: u)
                let cpu = cpuExpected(uniform: u)
                #expect(gpu == cpu, "uniform=\(u): gpu=\(gpu) cpu=\(cpu)")
            }
        }
    }

    @Test("GPU sample: large vocab + sparse peak (model-shaped)", .disabled("bisect"))
    func gpuSampleLargeVocab() {
        autoreleasepool {
            // Real models have vocab ~128-152K with a sparse peak.
            // Reproduce that shape to flush out any precision/overflow bugs.
            let n = 152_000
            var vals = [Float](repeating: -10.0, count: n)
            // Three tokens have most of the mass.
            vals[42] = 8.0  // dominant
            vals[1000] = 6.0  // secondary
            vals[50_000] = 4.0  // tertiary
            let l = Tensor.empty(shape: [n], dtype: .f32)
            l.copyIn(from: vals)

            // CPU reference (same algorithm as the kernel).
            let T: Float = 1.0
            let invT = 1.0 / T
            let scaled = vals.map { $0 * invT }
            let maxL = scaled.max() ?? 0
            let expSums = scaled.map { Foundation.exp(Double($0 - maxL)) }
            let total = expSums.reduce(0, +)

            // Sweep uniform draws — each must match the CPU CDF walk.
            for u in stride(from: Float(0.001), to: 1.0, by: 0.07) {
                let gpu = runGPUSample(logits: l, temperature: T, uniform: u)

                let target = Double(u) * total
                var cum = 0.0
                var expected = n - 1
                for i in 0 ..< n {
                    cum += expSums[i]
                    if cum >= target {
                        expected = i
                        break
                    }
                }
                #expect(gpu == expected, "uniform=\(u): gpu=\(gpu) cpu=\(expected)")
            }
        }
    }

    @Test("GPU sample: works on bf16 + f16 logits", .disabled("bisect"))
    func gpuSampleDtypes() {
        autoreleasepool {
            let f16 = Tensor.empty(shape: [4], dtype: .f16)
            f16.copyIn(from: [Float16(0), 0, 8, 0])  // peak at index 2
            #expect(runGPUSample(logits: f16, temperature: 1.0, uniform: 0.5) == 2)

            let bf16 = Tensor.empty(shape: [4], dtype: .bf16)
            // bf16 representations of 0, 0, 8, 0:
            bf16.copyIn(from: [UInt16(0), 0, 0x4100, 0])  // 0x4100 = bf16(8.0)
            #expect(runGPUSample(logits: bf16, temperature: 1.0, uniform: 0.5) == 2)
        }
    }

    // MARK: - argmax (GPU reduction kernel)

    /// Direct test of `Ops.argmax`. The GPU reduction kernel must return
    /// the index of the largest logit. Uses a vocab-realistic length so
    /// the 256-thread reduction spans many elements per thread.
    @Test("argmax f32 — GPU reduction returns index of the largest logit")
    func argmaxF32() {
        autoreleasepool {
            // Production-realistic logits length (covers the per-thread
            // strided scan in the 256-thread reduction).
            let n = 4096
            var logits = [Float](repeating: 0, count: n)
            for i in 0 ..< n { logits[i] = Float((i * 31) % 997) * 0.001 }
            let peak = 2718
            logits[peak] = 99.0  // unambiguous maximum
            let cpuArgmax = logits.indices.max(by: { logits[$0] < logits[$1] })!

            let logitsT = Tensor.empty(shape: [n], dtype: .f32)
            logitsT.copyIn(from: logits)
            let out = Tensor.empty(shape: [1], dtype: .u32)
            runAndWait { cb in Ops.argmax(logitsT, into: out, on: cb) }
            #expect(Int(out.toArray(as: UInt32.self)[0]) == cpuArgmax)
            #expect(Int(out.toArray(as: UInt32.self)[0]) == peak)
        }
    }

    /// argmax over f16/bf16 logits — exercises the half-precision kernel
    /// variants. The peak is large enough to survive bf16's 8-bit mantissa.
    @Test("argmax f16/bf16 — half-precision reduction returns the peak index")
    func argmaxHalfPrecision() {
        autoreleasepool {
            let n = 512

            let f16 = Tensor.empty(shape: [n], dtype: .f16)
            // Ramp of small values, one clear peak at index 300.
            var f16Data = (0 ..< n).map { Float16(Float($0 % 7) * 0.5) }
            f16Data[300] = 64.0
            f16.copyIn(from: f16Data)
            let outF16 = Tensor.empty(shape: [1], dtype: .u32)
            runAndWait { cb in Ops.argmax(f16, into: outF16, on: cb) }
            #expect(Int(outF16.toArray(as: UInt32.self)[0]) == 300)

            let bf16 = Tensor.empty(shape: [n], dtype: .bf16)
            // bf16 bits: ramp of small values, one peak (0x4280 = bf16(64.0)).
            var bf16Bits = [UInt16](repeating: 0x3F00, count: n)  // bf16(0.5)
            bf16Bits[123] = 0x4280  // bf16(64.0)
            bf16.copyIn(from: bf16Bits)
            let outBf16 = Tensor.empty(shape: [1], dtype: .u32)
            runAndWait { cb in Ops.argmax(bf16, into: outBf16, on: cb) }
            #expect(Int(outBf16.toArray(as: UInt32.self)[0]) == 123)
        }
    }

    // MARK: - softplus

    /// Direct test of `Ops.softplus`: out[i] = log(1 + exp(x[i])).
    @Test("softplus f32 — out[i] = log(1 + exp(x[i]))")
    func softplusF32() {
        autoreleasepool {
            let xs: [Float] = [0, 1, -1, 2, -5, 8, -20]
            let x = Tensor.empty(shape: [xs.count], dtype: .f32)
            x.copyIn(from: xs)
            var out: Tensor!
            runAndWait { cb in out = Ops.softplus(x, on: cb) }
            let r = out.toArray(as: Float.self)
            for i in 0 ..< xs.count {
                // Numerically-stable CPU reference: softplus(x) =
                // max(x,0) + log1p(exp(-|x|)).
                let v = xs[i]
                let expected = max(v, 0) + Float(log1p(Double(exp(-abs(v)))))
                #expect(
                    abs(r[i] - expected) < 1e-3,
                    "i=\(i) x=\(v) got \(r[i]) expected \(expected)")
            }
        }
    }

    // MARK: - rmsNormRows (multi-row RMSNorm reduction kernel)

    /// Direct test of `Ops.rmsNormRows`. Each of `nRows` rows is
    /// independently normalized: y = x / rms(x) * weight. The kernel is
    /// reduction-mode — rowSize must satisfy the rms_norm dispatch
    /// invariant (multiple of 128; one threadgroup per row).
    @Test("rmsNormRows f32 — each row independently normalized")
    func rmsNormRowsF32() {
        autoreleasepool {
            // rowSize=128 is the smallest legal size (32-lane simdgroup ×
            // 4 elements/thread → TPG = rowSize/4 = 32). See Ops.rmsNorm /
            // mlx/rms_norm.rs DISPATCH INVARIANTS.
            let nRows = 3
            let rowSize = 128
            let eps: Float = 1e-6

            // Distinct data per row so a row-offset bug would surface.
            var xData = [Float](repeating: 0, count: nRows * rowSize)
            for row in 0 ..< nRows {
                for d in 0 ..< rowSize {
                    xData[row * rowSize + d] = Float(d + 1) * Float(row + 1) * 0.1
                }
            }
            // Non-uniform weight so the weight multiply is exercised.
            let wData: [Float] = (0 ..< rowSize).map { 1.0 + Float($0 % 5) * 0.1 }

            let x = Tensor.empty(shape: [nRows, rowSize], dtype: .f32)
            x.copyIn(from: xData)
            let w = Tensor.empty(shape: [rowSize], dtype: .f32)
            w.copyIn(from: wData)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.rmsNormRows(
                    x, weight: w, eps: eps,
                    nRows: nRows, rowSize: rowSize, on: cb)
            }
            let r = out.toArray(as: Float.self)

            for row in 0 ..< nRows {
                var ssq: Float = 0
                for d in 0 ..< rowSize {
                    let v = xData[row * rowSize + d]
                    ssq += v * v
                }
                let rms = (ssq / Float(rowSize) + eps).squareRoot()
                for d in 0 ..< rowSize {
                    let expected = xData[row * rowSize + d] / rms * wData[d]
                    let got = r[row * rowSize + d]
                    #expect(
                        abs(got - expected) < 1e-2,
                        "row=\(row) d=\(d) got \(got) expected \(expected)")
                }
            }
        }
    }

    @Test("addAndRmsNorm — residual = a+b, normed = rmsNorm(a+b, weight)")
    func addAndRmsNormCorrectness() {
        autoreleasepool {
            // Two-row case so we exercise both the nRows=1 and
            // nRows>1 dispatch paths in one test. n=128 is the
            // smallest legal width (TPG = n/4 = 32 = simdgroup).
            let nRows = 2
            let n = 128
            let eps: Float = 1e-6
            let aData: [Float] = (0 ..< nRows * n).map { Float(($0 % 19) - 9) * 0.31 }
            let bData: [Float] = (0 ..< nRows * n).map { Float(($0 % 13) - 6) * 0.17 }
            let wData: [Float] = (0 ..< n).map { 1.0 + Float($0 % 7) * 0.05 }

            let a = Tensor.empty(shape: [nRows, n], dtype: .f32)
            a.copyIn(from: aData)
            let b = Tensor.empty(shape: [nRows, n], dtype: .f32)
            b.copyIn(from: bData)
            let weight = Tensor.empty(shape: [n], dtype: .f32)
            weight.copyIn(from: wData)

            let cmd = Device.shared.makeCommandBuffer()
            let (residual, normed) = Ops.addAndRmsNorm(
                a, b, weight: weight, eps: eps,
                nRows: nRows, rowSize: n, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            let residArr = residual.toArray(as: Float.self)
            let normedArr = normed.toArray(as: Float.self)

            // CPU reference: residual is a+b; normed is RMSNorm(a+b)·w.
            for row in 0 ..< nRows {
                var ssq: Float = 0
                var sums = [Float](repeating: 0, count: n)
                for d in 0 ..< n {
                    let s = aData[row * n + d] + bData[row * n + d]
                    sums[d] = s
                    ssq += s * s
                }
                let rms = (ssq / Float(n) + eps).squareRoot()
                for d in 0 ..< n {
                    let expectedResid = sums[d]
                    let expectedNormed = sums[d] / rms * wData[d]
                    let gotResid = residArr[row * n + d]
                    let gotNormed = normedArr[row * n + d]
                    #expect(
                        abs(gotResid - expectedResid) < 1e-3,
                        "residual row=\(row) d=\(d): got \(gotResid) expected \(expectedResid)")
                    #expect(
                        abs(gotNormed - expectedNormed) < 1e-2,
                        "normed row=\(row) d=\(d): got \(gotNormed) expected \(expectedNormed)")
                }
            }
        }
    }

    @Test("sdpaMulti — uniform K gives uniform attention (output = mean V)")
    func sdpaMultiUniformKMeansV() {
        autoreleasepool {
            // With every K row identical, all scores tie → softmax is
            // uniform → each query's output is the plain mean of the
            // attended V rows. A reference that needs no SDPA oracle.
            let headDim = 128
            let nQHeads = 2
            let nKVHeads = 1
            let baseKV = 0
            let nQuery = 4
            let kvStride = baseKV + nQuery
            let scale = 1.0 / Float(Double(headDim).squareRoot())

            let q = Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: .f32)
            q.copyIn(from: (0 ..< nQuery * nQHeads * headDim).map { Float($0 % 7) * 0.1 })

            // K: every row the same constant vector.
            let k = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            k.copyIn(from: [Float](repeating: 0.5, count: nKVHeads * kvStride * headDim))

            // V: row t holds the constant value `t` so the mean is easy.
            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * headDim)
            for t in 0 ..< kvStride {
                for d in 0 ..< headDim { vData[t * headDim + d] = Float(t) }
            }
            let v = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            v.copyIn(from: vData)

            let cmd = Device.shared.makeCommandBuffer()
            // Full (non-causal) mode → every query attends all 4 V rows,
            // so each output element is mean(0,1,2,3) = 1.5.
            let out = Ops.sdpaMulti(
                q: q, k: k, v: v,
                nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: headDim,
                baseKV: baseKV, nQuery: nQuery, kvStride: kvStride,
                causal: false, scale: scale, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            let result = out.toArray(as: Float.self)
            #expect(result.count == nQuery * nQHeads * headDim)
            for value in result {
                #expect(abs(value - 1.5) < 1e-3, "expected mean V = 1.5, got \(value)")
            }
        }
    }

    @Test("sdpaDecode2Pass f32 — blocks=32 matches single-pass sdpaDecode")
    func sdpaDecode2PassMatchesSinglePass() {
        autoreleasepool {
            // pass2 hardcodes a 32-lane block reduction, so blocks
            // must be ≥ 32 and a multiple of 32 — anything smaller
            // silently emits zero output.
            let D = 128
            let blocks = 32
            let kvStride = 64
            let nKV = 64
            let nQHeads = 2
            let nKVHeads = 1
            let q = Tensor.empty(shape: [nQHeads, D], dtype: .f32)
            var qData = [Float](repeating: 0, count: nQHeads * D)
            for h in 0 ..< nQHeads {
                qData[h * D] = Float(h + 1)
                qData[h * D + 1] = 0.5
            }
            q.copyIn(from: qData)
            let k = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)
            var kData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            for p in 0 ..< nKV {
                kData[p * D] = Float(p + 1) * 0.05
                kData[p * D + 1] = 0.1
            }
            k.copyIn(from: kData)
            let v = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)
            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            for p in 0 ..< nKV {
                for d in 0 ..< D {
                    vData[p * D + d] = Float((p + 1) * (d + 1)) * 0.01
                }
            }
            v.copyIn(from: vData)
            // Reference.
            var refOut: Tensor!
            runAndWait { cb in
                refOut = Ops.sdpaDecode(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads,
                    headDim: D, nKV: nKV, kvStride: kvStride,
                    scale: 1.0, on: cb)
            }
            // 2-pass.
            let partialO = Tensor.empty(shape: [nQHeads, blocks, D], dtype: .f32)
            let partialM = Tensor.empty(shape: [nQHeads, blocks], dtype: .f32)
            let partialL = Tensor.empty(shape: [nQHeads, blocks], dtype: .f32)
            let out = Tensor.empty(shape: [nQHeads, D], dtype: .f32)
            partialO.zero()
            partialM.zero()
            partialL.zero()
            out.zero()
            runAndWait { cb in
                Ops.sdpaDecode2Pass(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: D,
                    nKV: nKV, kvStride: kvStride, blocks: blocks,
                    scale: 1.0,
                    partialO: partialO, partialM: partialM, partialL: partialL,
                    into: out, on: cb)
            }
            let r = out.toArray(as: Float.self)
            let ref = refOut.toArray(as: Float.self)
            for i in 0 ..< nQHeads * D {
                #expect(abs(r[i] - ref[i]) < 1e-2, "i=\(i): \(r[i]) vs \(ref[i])")
            }
        }
    }

    @Test("sdpaMultiTreeMask — all-zero mask reproduces non-causal sdpaMulti output")
    func sdpaMultiTreeMaskAllowAll() {
        autoreleasepool {
            // An all-zero additive mask permits every position →
            // attention behaves exactly like causal=false sdpaMulti.
            let headDim = 128
            let nQHeads = 2
            let nKVHeads = 1
            let baseKV = 0
            let nQuery = 4
            let kvStride = baseKV + nQuery
            let scale = 1.0 / Float(Double(headDim).squareRoot())
            let q = Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: .f32)
            q.copyIn(from: (0 ..< nQuery * nQHeads * headDim).map { Float($0 % 7) * 0.1 })
            let k = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            k.copyIn(from: [Float](repeating: 0.5, count: nKVHeads * kvStride * headDim))
            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * headDim)
            for t in 0 ..< kvStride {
                for d in 0 ..< headDim { vData[t * headDim + d] = Float(t) }
            }
            let v = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            v.copyIn(from: vData)
            // Reference: plain non-causal sdpaMulti.
            var refOut: Tensor!
            runAndWait { cb in
                refOut = Ops.sdpaMulti(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: headDim,
                    baseKV: baseKV, nQuery: nQuery, kvStride: kvStride,
                    causal: false, scale: scale, on: cb)
            }
            // Tree-mask path: all-zero mask = permit everywhere.
            let mask = Tensor.empty(shape: [nQuery, nQuery], dtype: .f32)
            mask.copyIn(from: [Float](repeating: 0, count: nQuery * nQuery))
            var got: Tensor!
            runAndWait { cb in
                got = Ops.sdpaMultiTreeMask(
                    q: q, k: k, v: v, mask: mask,
                    nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: headDim,
                    baseKV: baseKV, nQuery: nQuery, kvStride: kvStride,
                    scale: scale, on: cb)
            }
            let r = refOut.toArray(as: Float.self)
            let g = got.toArray(as: Float.self)
            for i in 0 ..< r.count {
                #expect(abs(r[i] - g[i]) < 1e-3, "i=\(i): \(g[i]) vs \(r[i])")
            }
        }
    }

    @Test("sdpaMultiTreeMask — diagonal -inf mask isolates each query to its own row")
    func sdpaMultiTreeMaskDiagonalOnly() {
        autoreleasepool {
            // Mask = -inf off the diagonal, 0 on the diagonal → each
            // query attends only its own KV slot (assuming the cache
            // is full and queries are appended at the end). Outputs
            // collapse to a copy of the matching V row.
            let headDim = 128
            let nQHeads = 1
            let nKVHeads = 1
            let baseKV = 0
            let nQuery = 4
            let kvStride = baseKV + nQuery
            let scale = 1.0 / Float(Double(headDim).squareRoot())
            // Make every Q row identical and every K row identical so
            // attention only depends on the mask, not the dot products.
            let q = Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: .f32)
            q.copyIn(from: [Float](repeating: 0.5, count: nQuery * nQHeads * headDim))
            let k = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            k.copyIn(from: [Float](repeating: 0.5, count: nKVHeads * kvStride * headDim))
            // V row t holds the constant value `t`.
            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * headDim)
            for t in 0 ..< kvStride {
                for d in 0 ..< headDim { vData[t * headDim + d] = Float(t + 1) }
            }
            let v = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            v.copyIn(from: vData)
            let neg: Float = -1e9
            var maskData = [Float](repeating: neg, count: nQuery * nQuery)
            for i in 0 ..< nQuery { maskData[i * nQuery + i] = 0 }
            let mask = Tensor.empty(shape: [nQuery, nQuery], dtype: .f32)
            mask.copyIn(from: maskData)
            var got: Tensor!
            runAndWait { cb in
                got = Ops.sdpaMultiTreeMask(
                    q: q, k: k, v: v, mask: mask,
                    nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: headDim,
                    baseKV: baseKV, nQuery: nQuery, kvStride: kvStride,
                    scale: scale, on: cb)
            }
            let g = got.toArray(as: Float.self)
            for qIdx in 0 ..< nQuery {
                let expected = Float(qIdx + 1)
                for d in 0 ..< headDim {
                    #expect(
                        abs(g[qIdx * headDim + d] - expected) < 1e-3,
                        "q=\(qIdx) d=\(d): \(g[qIdx * headDim + d]) vs \(expected)")
                }
            }
        }
    }

    @Test("sdpaMulti — head_dim 256 routes to d256 kernel (uniform K → mean V)")
    func sdpaMultiHeadDim256UniformKMeansV() {
        autoreleasepool {
            // Identical to the d=128 uniform-K test but at head_dim=256
            // (Qwen3.6-A3B full-attention layers). Verifies the d256
            // routing exists and the kernel produces the same uniform-
            // attention result as the d128 path.
            let headDim = 256
            let nQHeads = 2
            let nKVHeads = 1
            let baseKV = 0
            let nQuery = 4
            let kvStride = baseKV + nQuery
            let scale = 1.0 / Float(Double(headDim).squareRoot())

            let q = Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: .f32)
            q.copyIn(from: (0 ..< nQuery * nQHeads * headDim).map { Float($0 % 7) * 0.1 })

            let k = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            k.copyIn(from: [Float](repeating: 0.5, count: nKVHeads * kvStride * headDim))

            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * headDim)
            for t in 0 ..< kvStride {
                for d in 0 ..< headDim { vData[t * headDim + d] = Float(t) }
            }
            let v = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            v.copyIn(from: vData)

            let cmd = Device.shared.makeCommandBuffer()
            let out = Ops.sdpaMulti(
                q: q, k: k, v: v,
                nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: headDim,
                baseKV: baseKV, nQuery: nQuery, kvStride: kvStride,
                causal: false, scale: scale, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            let result = out.toArray(as: Float.self)
            #expect(result.count == nQuery * nQHeads * headDim)
            for value in result {
                #expect(abs(value - 1.5) < 1e-3, "expected mean V = 1.5, got \(value)")
            }
        }
    }

    @Test("sdpaMulti — causal mode: query r attends V rows 0...r")
    func sdpaMultiCausalPrefixMeans() {
        autoreleasepool {
            let headDim = 128
            let nQHeads = 1
            let nKVHeads = 1
            let baseKV = 0
            let nQuery = 4
            let kvStride = baseKV + nQuery
            let scale = 1.0 / Float(Double(headDim).squareRoot())

            let q = Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: .f32)
            q.copyIn(from: [Float](repeating: 0.3, count: nQuery * nQHeads * headDim))
            let k = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            k.copyIn(from: [Float](repeating: 0.5, count: nKVHeads * kvStride * headDim))
            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * headDim)
            for t in 0 ..< kvStride {
                for d in 0 ..< headDim { vData[t * headDim + d] = Float(t) }
            }
            let v = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            v.copyIn(from: vData)

            let cmd = Device.shared.makeCommandBuffer()
            let out = Ops.sdpaMulti(
                q: q, k: k, v: v,
                nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: headDim,
                baseKV: baseKV, nQuery: nQuery, kvStride: kvStride,
                causal: true, scale: scale, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            // Causal: query r attends rows 0...r → output = mean(0...r).
            let result = out.toArray(as: Float.self)
            for r in 0 ..< nQuery {
                let expected = Float(r) / 2.0  // mean(0,1,...,r)
                for d in 0 ..< headDim {
                    let got = result[r * headDim + d]
                    #expect(
                        abs(got - expected) < 1e-3,
                        "query \(r) d=\(d): expected \(expected), got \(got)")
                }
            }
        }
    }

    @Test("sdpaBidirectional — d32/d64/d72 uniform K give mean V (no-causal contract)")
    func sdpaBidirectionalUniformKMeansV() {
        autoreleasepool {
            // Vision-tower contract: every query attends every key.
            // With identical K rows, softmax is uniform → each output
            // equals the plain mean of V across the attended block.
            // Cover every supported head_dim so the routing in
            // Ops.sdpaBidirectional is exercised end-to-end.
            for headDim in [32, 64, 72] {
                let nQHeads = 2
                let nKVHeads = 1
                let baseKV = 0
                let nQuery = 4
                let kvStride = baseKV + nQuery
                let scale = 1.0 / Float(Double(headDim).squareRoot())

                let q = Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: .f32)
                q.copyIn(from: (0 ..< nQuery * nQHeads * headDim).map { Float($0 % 7) * 0.1 })

                // K: every row the same constant vector → uniform scores.
                let k = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
                k.copyIn(
                    from: [Float](
                        repeating: 0.5,
                        count: nKVHeads * kvStride * headDim))

                // V: row t holds the constant value `t` so mean is easy.
                var vData = [Float](repeating: 0, count: nKVHeads * kvStride * headDim)
                for t in 0 ..< kvStride {
                    for d in 0 ..< headDim { vData[t * headDim + d] = Float(t) }
                }
                let v = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
                v.copyIn(from: vData)

                let cmd = Device.shared.makeCommandBuffer()
                let out = Ops.sdpaBidirectional(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads, headDim: headDim,
                    baseKV: baseKV, nQuery: nQuery, kvStride: kvStride,
                    scale: scale, on: cmd)
                cmd.commit()
                cmd.waitUntilCompleted()

                // Every query attends V rows {0,1,2,3} → mean = 1.5.
                let result = out.toArray(as: Float.self)
                #expect(result.count == nQuery * nQHeads * headDim)
                for value in result {
                    #expect(
                        abs(value - 1.5) < 1e-3,
                        "headDim=\(headDim): expected mean V = 1.5, got \(value)")
                }
            }
        }
    }

    @Test("sdpaDecode f32 — head_dim 512 (Gemma 4 global layer)")
    func sdpaDecodeD512() {
        autoreleasepool {
            // Gemma 4 global-attention layout: head_dim 512, 8 q-heads,
            // 1 KV head (GQA fan-out 8). Routes to the d512 kernel,
            // which dispatches at 512 threads/threadgroup (the 16-wide
            // per-lane footprint caps the pipeline below 1024).
            let D = 512
            let kvStride = 4
            let nKV = 1
            let nQHeads = 8
            let nKVHeads = 1

            let q = Tensor.empty(shape: [nQHeads, D], dtype: .f32)
            let k = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)
            let v = Tensor.empty(shape: [nKVHeads, kvStride, D], dtype: .f32)

            // Every q-head = e_0; K[0] = e_0 → dot(Q, K[0]) = 1.
            var qData = [Float](repeating: 0, count: nQHeads * D)
            for h in 0 ..< nQHeads { qData[h * D + 0] = 1 }
            q.copyIn(from: qData)

            var kData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            kData[0] = 1  // K[head=0, pos=0, d=0]
            k.copyIn(from: kData)

            // V[0] = ramp; positions [1..3] are zero so an over-read
            // past `n_kv` would show up in the output.
            var vData = [Float](repeating: 0, count: nKVHeads * kvStride * D)
            for d in 0 ..< D { vData[d] = Float(d + 1) }  // [1, 2, …, 512]
            v.copyIn(from: vData)

            var out: Tensor!
            runAndWait { cb in
                out = Ops.sdpaDecode(
                    q: q, k: k, v: v,
                    nQHeads: nQHeads, nKVHeads: nKVHeads,
                    headDim: D, nKV: nKV, kvStride: kvStride,
                    scale: 1.0, on: cb)
            }
            // n_kv = 1 → softmax([single_score]) = 1 → output == V[0]
            // for every q-head.
            let r = out.toArray(as: Float.self)
            for h in 0 ..< nQHeads {
                for d in 0 ..< D {
                    #expect(
                        abs(r[h * D + d] - Float(d + 1)) < 1e-3,
                        "head \(h) out[\(d)] = \(r[h * D + d]), expected \(d + 1)")
                }
            }
        }
    }

    @Test("ropeYaRN — factor=1 collapses to plain RoPE")
    func ropeYaRNFactorOneIsPlainRope() {
        autoreleasepool {
            // factor=1 → interpolation == extrapolation → the YaRN ramp
            // is a no-op and the kernel reduces to plain RoPE. `.plain`
            // also has attn_factor 1, so it must match Ops.rope exactly.
            let headDim = 128
            let nHeads = 2
            let qk = Tensor.empty(shape: [nHeads, headDim], dtype: .f32)
            qk.copyIn(from: (0 ..< nHeads * headDim).map { Float($0 % 13) * 0.1 - 0.5 })

            let cmd = Device.shared.makeCommandBuffer()
            let plain = Ops.rope(
                qk, position: 64, headDim: headDim,
                thetaBase: 1_000_000, scaling: .none, on: cmd)
            let yarn = Ops.ropeYaRN(
                qk, position: 64, headDim: headDim,
                thetaBase: 1_000_000, yarn: .plain, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            let p = plain.toArray(as: Float.self)
            let y = yarn.toArray(as: Float.self)
            for i in 0 ..< p.count {
                #expect(abs(p[i] - y[i]) < 1e-5, "i=\(i): plain \(p[i]) vs yarn \(y[i])")
            }
        }
    }

    @Test("ropeYaRN — position 0 is identity")
    func ropeYaRNIdentityAtPositionZero() {
        autoreleasepool {
            // position 0 → theta 0 → cos 1 / sin 0; attn_factor 1 → the
            // rotation is the identity regardless of the YaRN band.
            let headDim = 128
            let nHeads = 2
            let input = (0 ..< nHeads * headDim).map { Float($0 % 7) * 0.1 }
            let qk = Tensor.empty(shape: [nHeads, headDim], dtype: .f32)
            qk.copyIn(from: input)

            let yarn = Ops.RoPEYaRN.from(
                headDim: headDim, thetaBase: 1_000_000,
                factor: 16, betaFast: 32, betaSlow: 1,
                originalMaxPosition: 16384)
            let cmd = Device.shared.makeCommandBuffer()
            let out = Ops.ropeYaRN(
                qk, position: 0, headDim: headDim,
                thetaBase: 1_000_000, yarn: yarn, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            let o = out.toArray(as: Float.self)
            for i in 0 ..< o.count {
                #expect(abs(o[i] - input[i]) < 1e-5, "i=\(i): expected \(input[i]), got \(o[i])")
            }
        }
    }

    @Test("RoPEYaRN.from derives a sane correction band")
    func ropeYaRNFromCorrectionBand() {
        // Nemotron-Labs-Diffusion params — the band must land inside
        // [0, headDim) with high > low.
        let y = Ops.RoPEYaRN.from(
            headDim: 128, thetaBase: 1_000_000,
            factor: 16, betaFast: 32, betaSlow: 1,
            originalMaxPosition: 16384)
        #expect(y.factor == 16)
        #expect(y.low >= 0 && y.low < y.high)
        #expect(y.high <= 127)
        #expect(y.attnFactor == 1)  // mscale == mscale_all_dim default
    }

    @Test("gemm — multi-row matmul matches a CPU reference")
    func gemmMatchesCPU() {
        autoreleasepool {
            // out[r,o] = Σ_k weight[o,k]·input[r,k]. nRows / outDim are
            // not multiples of the 32×32 tile — exercises the edge path.
            let nRows = 5
            let inDim = 48
            let outDim = 7
            let wData = (0 ..< outDim * inDim).map { Float($0 % 11) * 0.1 - 0.4 }
            let xData = (0 ..< nRows * inDim).map { Float($0 % 9) * 0.1 - 0.2 }
            let weight = Tensor.empty(shape: [outDim, inDim], dtype: .f32)
            let input = Tensor.empty(shape: [nRows, inDim], dtype: .f32)
            weight.copyIn(from: wData)
            input.copyIn(from: xData)

            let cmd = Device.shared.makeCommandBuffer()
            let out = Ops.gemm(weight: weight, input: input, nRows: nRows, on: cmd)
            cmd.commit()
            cmd.waitUntilCompleted()

            let got = out.toArray(as: Float.self)
            #expect(got.count == nRows * outDim)
            for r in 0 ..< nRows {
                for o in 0 ..< outDim {
                    var acc: Float = 0
                    for k in 0 ..< inDim { acc += wData[o * inDim + k] * xData[r * inDim + k] }
                    #expect(
                        abs(got[r * outDim + o] - acc) < 1e-4,
                        "r=\(r) o=\(o): expected \(acc), got \(got[r * outDim + o])")
                }
            }
        }
    }

    @Test("ropeProportional f32 — matches a CPU ProportionalRoPE oracle")
    func ropeProportionalMatchesOracle() {
        autoreleasepool {
            // Gemma 4 global-layer ProportionalRoPE: dims=512,
            // rotatedDim=128. Rotates pairs (i, i + 256) for i ∈ [0, 64)
            // at frequency theta^(-2i/512); dims [64, 256) and their
            // partners pass through untouched. Drives the shared
            // ffai_rope_llama kernel — this pins the pairing offset
            // (headDim/2, NOT rotatedDim/2) and the frequency
            // denominator (the full headDim).
            let headDim = 512
            let rotatedDim = 128
            let theta: Float = 1_000_000
            let position = 7
            let nHeads = 3

            // Deterministic, non-trivial input.
            var data = [Float](repeating: 0, count: nHeads * headDim)
            for i in 0 ..< data.count {
                data[i] = Float((i * 7) % 19 - 9) * 0.13
            }
            let qk = Tensor.empty(shape: [nHeads, headDim], dtype: .f32)
            qk.copyIn(from: data)

            runAndWait { cb in
                Gemma4Ops.ropeProportional(
                    qk, position: position, headDim: headDim,
                    rotatedDim: rotatedDim, thetaBase: theta, on: cb)
            }
            let got = qk.toArray(as: Float.self)

            // CPU oracle of the reference's ProportionalRoPE.
            let half = headDim / 2  // 256 — pairing offset
            let rotatedPairs = rotatedDim / 2  // 64 — rotated pair count
            var expected = data
            for h in 0 ..< nHeads {
                let base = h * headDim
                for i in 0 ..< rotatedPairs {
                    // inv_freq = theta^(-2i/headDim) = theta^(-i/half)
                    let invFreq = Float(
                        pow(Double(theta), -Double(i) / Double(half)))
                    let angle = Double(position) * Double(invFreq)
                    let c = Float(cos(angle))
                    let s = Float(sin(angle))
                    let x1 = data[base + i]
                    let x2 = data[base + i + half]
                    expected[base + i] = x1 * c - x2 * s
                    expected[base + i + half] = x1 * s + x2 * c
                }
                // i ∈ [rotatedPairs, half): pass-through (unchanged).
            }
            for idx in 0 ..< got.count {
                #expect(
                    abs(got[idx] - expected[idx]) < 1e-3,
                    "idx=\(idx): got \(got[idx]) expected \(expected[idx])")
            }
        }
    }

    @Test("clampScalar f32 — out[i] = min(max(x[i], lo), hi)")
    func clampScalarF32() {
        autoreleasepool {
            let x = Tensor.empty(shape: [6], dtype: .f32)
            x.copyIn(from: [Float(-10), -6.375, 0, 3, 6.312, 20])
            var out: Tensor!
            runAndWait { cb in
                out = Ops.clampScalar(x, lo: -6.375, hi: 6.312, on: cb)
            }
            let got = out.toArray(as: Float.self)
            let expected: [Float] = [-6.375, -6.375, 0, 3, 6.312, 6.312]
            for i in 0 ..< got.count {
                #expect(
                    abs(got[i] - expected[i]) < 1e-5,
                    "idx \(i): got \(got[i]) expected \(expected[i])")
            }
        }
    }

    @Test("resizeNormalize f32 — bilinear resize + per-channel normalize → NCHW")
    func resizeNormalizeF32() {
        autoreleasepool {
            let (sw, sh, tw, th) = (4, 3, 6, 5)
            let mean: (Float, Float, Float) = (0.5, 0.45, 0.4)
            let std: (Float, Float, Float) = (0.5, 0.5, 0.5)
            var px = [Float](repeating: 0, count: sh * sw * 3)
            for i in 0 ..< px.count { px[i] = Float(i % 13) / 13.0 }
            let img = RGBImage(width: sw, height: sh, pixels: px)
            let out = MediaUtils.resizeNormalize(
                img, targetW: tw, targetH: th,
                normalization: ImageNormalization(mean: mean, std: std))
            #expect(out.shape == [1, 3, th, tw])
            let got = out.toArray(as: Float.self)

            // CPU bilinear reference (align_corners=false), planar NCHW.
            let m = [mean.0, mean.1, mean.2]
            let s = [std.0, std.1, std.2]
            let scaleX = Float(sw) / Float(tw)
            let scaleY = Float(sh) / Float(th)
            func clampf(_ v: Float, _ hi: Float) -> Float { max(0, min(hi, v)) }
            var maxDiff: Float = 0
            for oy in 0 ..< th {
                let syc = clampf((Float(oy) + 0.5) * scaleY - 0.5, Float(sh - 1))
                let y0 = syc.rounded(.down)
                let wy = syc - y0
                let y0i = Int(y0)
                let y1i = Int(clampf(y0 + 1, Float(sh - 1)))
                for ox in 0 ..< tw {
                    let sxc = clampf((Float(ox) + 0.5) * scaleX - 0.5, Float(sw - 1))
                    let x0 = sxc.rounded(.down)
                    let wx = sxc - x0
                    let x0i = Int(x0)
                    let x1i = Int(clampf(x0 + 1, Float(sw - 1)))
                    for c in 0 ..< 3 {
                        let p00 = px[(y0i * sw + x0i) * 3 + c]
                        let p01 = px[(y0i * sw + x1i) * 3 + c]
                        let p10 = px[(y1i * sw + x0i) * 3 + c]
                        let p11 = px[(y1i * sw + x1i) * 3 + c]
                        let top = p00 * (1 - wx) + p01 * wx
                        let bot = p10 * (1 - wx) + p11 * wx
                        let v = (top * (1 - wy) + bot * wy - m[c]) / s[c]
                        let idx = (c * th + oy) * tw + ox
                        maxDiff = max(maxDiff, abs(got[idx] - v))
                    }
                }
            }
            #expect(maxDiff < 1e-4, "resizeNormalize diverges (max \(maxDiff))")
        }
    }

    @Test("resizeNormalize f32 bicubic — Catmull-Rom (a=-0.75) resize + normalize → NCHW")
    func resizeNormalizeBicubicF32() {
        autoreleasepool {
            // Down-scale (the smart_resize regime bicubic targets): 8×6 → 5×4.
            let (sw, sh, tw, th) = (8, 6, 5, 4)
            let mean: (Float, Float, Float) = (0.5, 0.45, 0.4)
            let std: (Float, Float, Float) = (0.5, 0.5, 0.5)
            var px = [Float](repeating: 0, count: sh * sw * 3)
            for i in 0 ..< px.count { px[i] = Float(i % 17) / 17.0 }
            let img = RGBImage(width: sw, height: sh, pixels: px)
            let out = MediaUtils.resizeNormalize(
                img, targetW: tw, targetH: th,
                normalization: ImageNormalization(mean: mean, std: std),
                mode: .bicubic)
            #expect(out.shape == [1, 3, th, tw])
            let got = out.toArray(as: Float.self)

            // CPU bicubic reference — mirrors resize_normalize.rs `cpu_ref_bicubic`
            // exactly (a = -0.75, 4×4 tap, half-pixel sampling, edge clamp).
            let a: Float = -0.75
            func cubic(_ d: Float) -> Float {
                let ad = abs(d)
                if ad <= 1 { return (a + 2) * ad * ad * ad - (a + 3) * ad * ad + 1 }
                if ad < 2 { return a * ad * ad * ad - 5 * a * ad * ad + 8 * a * ad - 4 * a }
                return 0
            }
            func clampi(_ v: Int, _ hi: Int) -> Int { max(0, min(hi, v)) }
            let m = [mean.0, mean.1, mean.2]
            let s = [std.0, std.1, std.2]
            let scaleX = Float(sw) / Float(tw)
            let scaleY = Float(sh) / Float(th)
            var maxDiff: Float = 0
            for oy in 0 ..< th {
                let sy = (Float(oy) + 0.5) * scaleY - 0.5
                let fy = sy.rounded(.down)
                for ox in 0 ..< tw {
                    let sx = (Float(ox) + 0.5) * scaleX - 0.5
                    let fx = sx.rounded(.down)
                    for c in 0 ..< 3 {
                        var acc: Float = 0
                        for j in 0 ..< 4 {
                            let yj = Int(fy) + j - 1
                            let wy = cubic(sy - (fy + Float(j - 1)))
                            let yu = clampi(yj, sh - 1)
                            for i in 0 ..< 4 {
                                let xi = Int(fx) + i - 1
                                let wx = cubic(sx - (fx + Float(i - 1)))
                                let xu = clampi(xi, sw - 1)
                                acc += wy * wx * px[(yu * sw + xu) * 3 + c]
                            }
                        }
                        let v = (acc - m[c]) / s[c]
                        let idx = (c * th + oy) * tw + ox
                        maxDiff = max(maxDiff, abs(got[idx] - v))
                    }
                }
            }
            #expect(maxDiff < 1e-4, "resizeNormalize bicubic diverges (max \(maxDiff))")
        }
    }

    @Test(
        "transposeTokensToHeads f32 — [nTokens,nHeads,d] → [nHeads,nTokens,d] at production shape")
    func transposeTokensToHeadsF32() {
        autoreleasepool {
            // SigLIP production shape: 576 patches, 16 heads, head_dim 64.
            let nTokens = 576, nHeads = 16, headDim = 64
            let count = nTokens * nHeads * headDim
            var src = [Float](repeating: 0, count: count)
            for i in 0 ..< count { src[i] = Float((i % 29) - 14) * 0.1 }
            let x = Tensor.empty(shape: [nTokens, nHeads, headDim], dtype: .f32)
            x.copyIn(from: src)
            var out: Tensor!
            runAndWait { cb in
                out = Ops.transposeTokensToHeads(
                    x, nTokens: nTokens, nHeads: nHeads, headDim: headDim,
                    on: cb)
            }
            let got = out.toArray(as: Float.self)
            #expect(out.shape == [nHeads, nTokens, headDim])
            // CPU reference: out[(h*nTokens+t)*d + e] = src[(t*nHeads+h)*d + e].
            var mismatches = 0
            for t in 0 ..< nTokens {
                for h in 0 ..< nHeads {
                    for e in 0 ..< headDim {
                        let inIdx = (t * nHeads + h) * headDim + e
                        let outIdx = (h * nTokens + t) * headDim + e
                        if got[outIdx] != src[inIdx] { mismatches += 1 }
                    }
                }
            }
            #expect(mismatches == 0, "\(mismatches) transposed elements mismatched")
        }
    }

    @Test("sdpaBidirectionalRelPos — GPU matches a CPU naive reference (d128)")
    func sdpaBidirectionalRelPos() {
        autoreleasepool {
            // Conformer-style encoder self-attention: MHA, base_kv=0,
            // nQuery=nKv=L, headDim=128, per-head additive relpos bias.
            let headDim = 128
            let nQHeads = 4, nKVHeads = 4
            let baseKV = 0, nQuery = 20
            let kvStride = baseKV + nQuery
            let relZero = baseKV + nQuery - 1
            let relLen = baseKV + 2 * nQuery - 1
            let scale = 1.0 / Float(Double(headDim).squareRoot())
            let gqa = nQHeads / nKVHeads
            let nKV = baseKV + nQuery

            func ramp(_ n: Int, _ step: Float, _ start: Float) -> [Float] {
                (0 ..< n).map { (start + Float($0) * step).truncatingRemainder(dividingBy: 2.0) - 1.0 }
            }
            let q = ramp(nQuery * nQHeads * headDim, 0.013, -0.4)
            let k = ramp(nKVHeads * kvStride * headDim, 0.011, -0.5)
            let v = ramp(nKVHeads * kvStride * headDim, 0.007, -0.3)
            let bias = (0 ..< nQHeads * relLen).map { (Float($0 % 7) - 3.0) * 0.05 }

            // CPU naive reference (mirrors the metaltile oracle):
            // softmax_j( scale·Qᵢ·Kⱼ + bias[h, (j+relZero)-(baseKV+i)] )·Vⱼ.
            var expected = [Float](repeating: 0, count: nQuery * nQHeads * headDim)
            for r in 0 ..< nQuery {
                let qAbs = baseKV + r
                for qh in 0 ..< nQHeads {
                    let kvh = qh / gqa
                    let qOff = (r * nQHeads + qh) * headDim
                    let kvSlab = kvh * kvStride * headDim
                    var scores = [Float](repeating: 0, count: nKV)
                    for t in 0 ..< nKV {
                        let kOff = kvSlab + t * headDim
                        var dot: Float = 0
                        for d in 0 ..< headDim { dot += q[qOff + d] * k[kOff + d] }
                        let rel = min(max(0, t + relZero - qAbs), relLen - 1)
                        scores[t] = dot * scale + bias[qh * relLen + rel]
                    }
                    let m = scores.max() ?? 0
                    var sum: Float = 0
                    for t in 0 ..< nKV { scores[t] = exp(scores[t] - m); sum += scores[t] }
                    let inv = sum > 0 ? 1 / sum : 0
                    for d in 0 ..< headDim {
                        var acc: Float = 0
                        for t in 0 ..< nKV { acc += scores[t] * inv * v[kvSlab + t * headDim + d] }
                        expected[qOff + d] = acc
                    }
                }
            }

            let qT = Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: .f32)
            qT.copyIn(from: q)
            let kT = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            kT.copyIn(from: k)
            let vT = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            vT.copyIn(from: v)
            let biasT = Tensor.empty(shape: [nQHeads, relLen], dtype: .f32)
            biasT.copyIn(from: bias)

            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.sdpaBidirectionalRelPos(
                    q: qT, k: kT, v: vT, bias: biasT,
                    nQHeads: nQHeads, nKVHeads: nKVHeads,
                    baseKV: baseKV, nQuery: nQuery, kvStride: kvStride,
                    relZero: relZero, relLen: relLen, scale: scale, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            #expect(got.count == expected.count)
            var maxDiff: Float = 0
            for i in 0 ..< min(got.count, expected.count) {
                maxDiff = max(maxDiff, abs(got[i] - expected[i]))
            }
            #expect(
                maxDiff < 1e-3,
                "sdpaBidirectionalRelPos GPU diverges from CPU reference (max |Δ| \(maxDiff))")
        }
    }

    @Test("sdpaRelPosConformer — GPU matches a CPU ESPnet rel-pos reference (d128)")
    func sdpaRelPosConformer() {
        autoreleasepool {
            // ESPnet RelPositionMultiHeadedAttention: score =
            // scale·( (Qᵢ+uₕ)·Kⱼ + (Qᵢ+vₕ)·relEmb[dist] ), bidirectional softmax.
            let headDim = 128
            let nQHeads = 4, nKVHeads = 2  // exercise GQA fan-out
            let baseKV = 0, nQuery = 12
            let kvStride = baseKV + nQuery
            let relZero = baseKV + nQuery - 1
            let relLen = baseKV + 2 * nQuery - 1
            let scale = 1.0 / Float(Double(headDim).squareRoot())
            let gqa = nQHeads / nKVHeads
            let nKV = baseKV + nQuery

            func ramp(_ n: Int, _ step: Float, _ start: Float) -> [Float] {
                (0 ..< n).map { (start + Float($0) * step).truncatingRemainder(dividingBy: 2.0) - 1.0 }
            }
            let q = ramp(nQuery * nQHeads * headDim, 0.013, -0.4)
            let k = ramp(nKVHeads * kvStride * headDim, 0.011, -0.5)
            let v = ramp(nKVHeads * kvStride * headDim, 0.007, -0.3)
            let relEmb = ramp(relLen * headDim, 0.005, -0.2)
            let posBiasU = ramp(nQHeads * headDim, 0.009, -0.1)
            let posBiasV = ramp(nQHeads * headDim, 0.006, 0.15)

            // CPU naive reference (mirrors sdpa_rel_pos_conformer.rs):
            // matrix_ac = (Q+u)·K, matrix_bd = (Q+v)·relEmb[dist].
            var expected = [Float](repeating: 0, count: nQuery * nQHeads * headDim)
            for r in 0 ..< nQuery {
                let qAbs = baseKV + r
                for qh in 0 ..< nQHeads {
                    let kvh = qh / gqa
                    let qOff = (r * nQHeads + qh) * headDim
                    let uOff = qh * headDim
                    let kvSlab = kvh * kvStride * headDim
                    var scores = [Float](repeating: 0, count: nKV)
                    for t in 0 ..< nKV {
                        let kOff = kvSlab + t * headDim
                        let rel = min(max(0, t + relZero - qAbs), relLen - 1)
                        let rOff = rel * headDim
                        var ac: Float = 0
                        var bd: Float = 0
                        for d in 0 ..< headDim {
                            ac += (q[qOff + d] + posBiasU[uOff + d]) * k[kOff + d]
                            bd += (q[qOff + d] + posBiasV[uOff + d]) * relEmb[rOff + d]
                        }
                        scores[t] = scale * (ac + bd)
                    }
                    let m = scores.max() ?? 0
                    var sum: Float = 0
                    for t in 0 ..< nKV { scores[t] = exp(scores[t] - m); sum += scores[t] }
                    let inv = sum > 0 ? 1 / sum : 0
                    for d in 0 ..< headDim {
                        var acc: Float = 0
                        for t in 0 ..< nKV { acc += scores[t] * inv * v[kvSlab + t * headDim + d] }
                        expected[qOff + d] = acc
                    }
                }
            }

            let qT = Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: .f32); qT.copyIn(from: q)
            let kT = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32); kT.copyIn(from: k)
            let vT = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32); vT.copyIn(from: v)
            let relT = Tensor.empty(shape: [relLen, headDim], dtype: .f32); relT.copyIn(from: relEmb)
            let uT = Tensor.empty(shape: [nQHeads, headDim], dtype: .f32); uT.copyIn(from: posBiasU)
            let vbT = Tensor.empty(shape: [nQHeads, headDim], dtype: .f32); vbT.copyIn(from: posBiasV)

            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.sdpaRelPosConformer(
                    q: qT, k: kT, v: vT,
                    relEmb: relT, posBiasU: uT, posBiasV: vbT,
                    nQHeads: nQHeads, nKVHeads: nKVHeads,
                    baseKV: baseKV, nQuery: nQuery, kvStride: kvStride,
                    relZero: relZero, relLen: relLen, scale: scale, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            #expect(got.count == expected.count)
            var maxDiff: Float = 0
            for i in 0 ..< min(got.count, expected.count) {
                maxDiff = max(maxDiff, abs(got[i] - expected[i]))
            }
            #expect(
                maxDiff < 1e-3,
                "sdpaRelPosConformer GPU diverges from CPU reference (max |Δ| \(maxDiff))")
        }
    }

    @Test("sdpaRelPosConformer — row-reversed table + zero bias reproduces a GraniteSpeech block")
    func sdpaRelPosConformerGraniteMapping() {
        autoreleasepool {
            // GraniteSpeech's conformer scores are scale·( qᵢ·kⱼ +
            // qᵢ·relEmb[(i−j)+maxPos] ) with bidirectional softmax — ESPnet
            // rel-pos WITHOUT the u/v content biases, and with the OPPOSITE
            // distance sign to the kernel (which indexes j−i+relZero). The
            // GPU path therefore feeds a ROW-REVERSED embedding table and
            // relZero = maxPos with zero u/v biases. This pins that the
            // mapping reproduces the GraniteSpeech formula exactly, so the
            // encoder wiring in GraniteSpeech.swift is provably equivalent
            // to the CPU path it replaces.
            let H = 2, C = 8, Dh = 128
            let maxPos = C  // ≥ contextSize, mirroring the real config
            let relLen = 2 * maxPos + 1
            let scale = 1.0 / Float(Double(Dh).squareRoot())

            func ramp(_ n: Int, _ step: Float, _ start: Float) -> [Float] {
                (0 ..< n).map {
                    (start + Float($0) * step).truncatingRemainder(dividingBy: 2.0) - 1.0
                }
            }
            // Head-major blocks [H][C][Dh], as conformerAttn extracts them.
            func block(_ seed: Float) -> [[[Float]]] {
                var b = [[[Float]]](
                    repeating: [[Float]](repeating: [Float](repeating: 0, count: Dh), count: C),
                    count: H)
                var n: Float = seed
                for h in 0 ..< H {
                    for t in 0 ..< C {
                        for d in 0 ..< Dh {
                            b[h][t][d] = (n * 0.013).truncatingRemainder(dividingBy: 2.0) - 1.0
                            n += 1
                        }
                    }
                }
                return b
            }
            let qB = block(3), kB = block(17), vB = block(29)
            let relPosW = ramp(relLen * Dh, 0.004, -0.25)  // [relLen, Dh] granite table

            // ── CPU GraniteSpeech reference ──
            var expected = [[[Float]]](
                repeating: [[Float]](repeating: [Float](repeating: 0, count: Dh), count: C),
                count: H)
            for h in 0 ..< H {
                for i in 0 ..< C {
                    var scores = [Float](repeating: 0, count: C)
                    for j in 0 ..< C {
                        var ac: Float = 0, bd: Float = 0
                        let dist = (i - j) + maxPos
                        let rOff = dist * Dh
                        for d in 0 ..< Dh {
                            ac += qB[h][i][d] * kB[h][j][d]
                            bd += qB[h][i][d] * relPosW[rOff + d]
                        }
                        scores[j] = scale * (ac + bd)
                    }
                    let m = scores.max() ?? 0
                    var sum: Float = 0
                    for j in 0 ..< C { scores[j] = exp(scores[j] - m); sum += scores[j] }
                    let inv = sum > 0 ? 1 / sum : 0
                    for d in 0 ..< Dh {
                        var acc: Float = 0
                        for j in 0 ..< C { acc += scores[j] * inv * vB[h][j][d] }
                        expected[h][i][d] = acc
                    }
                }
            }

            // ── GPU via row-reversed table + zero u/v bias ──
            let device = Device.shared
            var qFlat = [Float](repeating: 0, count: C * H * Dh)  // [C, H, Dh]
            var kFlat = [Float](repeating: 0, count: H * C * Dh)  // [H, C, Dh]
            var vFlat = [Float](repeating: 0, count: H * C * Dh)
            for h in 0 ..< H {
                for t in 0 ..< C {
                    let qDst = (t * H + h) * Dh, kvDst = (h * C + t) * Dh
                    for d in 0 ..< Dh {
                        qFlat[qDst + d] = qB[h][t][d]
                        kFlat[kvDst + d] = kB[h][t][d]
                        vFlat[kvDst + d] = vB[h][t][d]
                    }
                }
            }
            var flipped = [Float](repeating: 0, count: relLen * Dh)
            for r in 0 ..< relLen {
                let src = (relLen - 1 - r) * Dh, dst = r * Dh
                for d in 0 ..< Dh { flipped[dst + d] = relPosW[src + d] }
            }
            let qT = Tensor.empty(shape: [C, H, Dh], dtype: .f32, device: device); qT.copyIn(from: qFlat)
            let kT = Tensor.empty(shape: [H, C, Dh], dtype: .f32, device: device); kT.copyIn(from: kFlat)
            let vT = Tensor.empty(shape: [H, C, Dh], dtype: .f32, device: device); vT.copyIn(from: vFlat)
            let relT = Tensor.empty(shape: [relLen, Dh], dtype: .f32, device: device)
            relT.copyIn(from: flipped)
            let zeroT = Tensor.empty(shape: [H, Dh], dtype: .f32, device: device)
            zeroT.copyIn(from: [Float](repeating: 0, count: H * Dh))

            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.sdpaRelPosConformer(
                    q: qT, k: kT, v: vT, relEmb: relT, posBiasU: zeroT, posBiasV: zeroT,
                    nQHeads: H, nKVHeads: H, baseKV: 0, nQuery: C, kvStride: C,
                    relZero: maxPos, relLen: relLen, scale: scale, on: cmd)
            }
            let got = outT.toArray(as: Float.self)  // [C, H, Dh]
            var maxDiff: Float = 0
            for t in 0 ..< C {
                for h in 0 ..< H {
                    let src = (t * H + h) * Dh
                    for d in 0 ..< Dh {
                        maxDiff = max(maxDiff, abs(got[src + d] - expected[h][t][d]))
                    }
                }
            }
            #expect(
                maxDiff < 1e-3,
                "GraniteSpeech rel-pos mapping diverges from CPU (max |Δ| \(maxDiff))")
        }
    }

    @Test("melSpectrogramMagnitude — GPU matches a CPU magnitude-DFT reference")
    func melSpectrogramMagnitudeF32() {
        autoreleasepool {
            // Small STFT: nFFT=8 → nFreq=5, hop=2, 3 mel bins, 4 frames.
            let nFFT = 8, nFreq = nFFT / 2 + 1, nMels = 3, hop = 2, nFrames = 4
            let logEps: Float = 1e-3
            let nSamples = (nFrames - 1) * hop + nFFT  // exact pre-pad

            func ramp(_ n: Int, _ seed: Int, _ amp: Float) -> [Float] {
                (0 ..< n).map { Float(($0 * 7 + seed) % 13) / 13 * amp - amp / 2 }
            }
            let audio = ramp(nSamples, 1, 1.0)
            // Periodic Hann window.
            let window = (0 ..< nFFT).map {
                0.5 - 0.5 * Foundation.cos(2 * Float.pi * Float($0) / Float(nFFT))
            }
            let melW = ramp(nMels * nFreq, 4, 0.5).map { abs($0) + 0.05 }  // non-negative filterbank

            // CPU oracle — MAGNITUDE DFT: mag[k] = sqrt(re²+im²), then
            // log(mel @ mag + eps).
            var expected = [Float](repeating: 0, count: nFrames * nMels)
            for f in 0 ..< nFrames {
                let base = f * hop
                var mag = [Float](repeating: 0, count: nFreq)
                for kbin in 0 ..< nFreq {
                    var re: Float = 0, im: Float = 0
                    let a = -2 * Float.pi * Float(kbin) / Float(nFFT)
                    for t in 0 ..< nFFT {
                        let s = audio[base + t] * window[t]
                        re += s * Foundation.cos(a * Float(t))
                        im += s * Foundation.sin(a * Float(t))
                    }
                    mag[kbin] = (re * re + im * im).squareRoot()
                }
                for m in 0 ..< nMels {
                    var acc: Float = 0
                    for kbin in 0 ..< nFreq { acc += melW[m * nFreq + kbin] * mag[kbin] }
                    expected[f * nMels + m] = Foundation.log(acc + logEps)
                }
            }

            let audioT = Tensor.empty(shape: [nSamples], dtype: .f32); audioT.copyIn(from: audio)
            let winT = Tensor.empty(shape: [nFFT], dtype: .f32); winT.copyIn(from: window)
            let melT = Tensor.empty(shape: [nMels, nFreq], dtype: .f32); melT.copyIn(from: melW)

            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.melSpectrogramMagnitude(
                    audio: audioT, window: winT, melWeight: melT,
                    nFFT: nFFT, nMels: nMels, hopLength: hop,
                    nFrames: nFrames, logEps: logEps, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            var maxDiff: Float = 0
            for i in 0 ..< expected.count { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-3, "melSpectrogramMagnitude diverges from CPU oracle (max \(maxDiff))")
        }
    }

    @Test("avgPool2dNHWC — GPU matches a CPU NHWC reference (2×2 s2, count_include_pad=false)")
    func avgPool2dNHWCF32() {
        autoreleasepool {
            let (batch, inH, inW, ch) = (2, 5, 4, 3)
            let (kH, kW, stride, pad) = (2, 2, 2, 1)  // pad → edge windows divide by real taps
            let outH = (inH + 2 * pad - kH) / stride + 1
            let outW = (inW + 2 * pad - kW) / stride + 1
            let input = (0 ..< batch * inH * inW * ch).map { Float(($0 * 3 + 1) % 19) / 19 }

            var expected = [Float](repeating: 0, count: batch * outH * outW * ch)
            for n in 0 ..< batch {
                for oh in 0 ..< outH {
                    for ow in 0 ..< outW {
                        for c in 0 ..< ch {
                            var acc: Float = 0
                            var count: Float = 0
                            for ky in 0 ..< kH {
                                let ph = oh * stride + ky
                                guard ph >= pad && ph < pad + inH else { continue }
                                let ih = ph - pad
                                for kx in 0 ..< kW {
                                    let pw = ow * stride + kx
                                    guard pw >= pad && pw < pad + inW else { continue }
                                    let iw = pw - pad
                                    acc += input[((n * inH + ih) * inW + iw) * ch + c]
                                    count += 1
                                }
                            }
                            let v = count > 0 ? acc / count : 0
                            expected[((n * outH + oh) * outW + ow) * ch + c] = v
                        }
                    }
                }
            }
            let inT = Tensor.empty(shape: [batch, inH, inW, ch], dtype: .f32); inT.copyIn(from: input)
            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.avgPool2dNHWC(
                    inT, batch: batch, inH: inH, inW: inW, channels: ch,
                    kH: kH, kW: kW, stride: stride, pad: pad, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            var maxDiff: Float = 0
            for i in 0 ..< expected.count { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-5, "avgPool2dNHWC diverges (max \(maxDiff))")
        }
    }

    @Test("im2colPatch — GPU matches a CPU NCHW unfold reference")
    func im2colPatchF32() {
        autoreleasepool {
            let (channels, patch) = (3, 2)
            let (gridH, gridW) = (3, 4)
            let (inH, inW) = (gridH * patch, gridW * patch)
            let nPatches = gridH * gridW
            let patchDim = channels * patch * patch
            let input = (0 ..< channels * inH * inW).map { Float(($0 * 5 + 2) % 23) / 23 }

            var expected = [Float](repeating: 0, count: nPatches * patchDim)
            for pr in 0 ..< gridH {
                for pc in 0 ..< gridW {
                    let p = pr * gridW + pc
                    for c in 0 ..< channels {
                        for py in 0 ..< patch {
                            for px in 0 ..< patch {
                                let col = (c * patch + py) * patch + px
                                let inY = pr * patch + py
                                let inX = pc * patch + px
                                expected[p * patchDim + col] = input[(c * inH + inY) * inW + inX]
                            }
                        }
                    }
                }
            }
            let inT = Tensor.empty(shape: [channels, inH, inW], dtype: .f32); inT.copyIn(from: input)
            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.im2colPatch(inT, channels: channels, inH: inH, inW: inW, patch: patch, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            #expect(outT.shape == [nPatches, patchDim])
            var maxDiff: Float = 0
            for i in 0 ..< expected.count { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-6, "im2colPatch diverges (max \(maxDiff))")
        }
    }

    @Test("posEmb2dAdd — GPU matches a CPU factorized 2D pos-emb reference")
    func posEmb2dAddF32() {
        autoreleasepool {
            let (hidden, gridW, gridH) = (8, 4, 3)
            let nPatches = gridH * gridW
            let tokens = (0 ..< nPatches * hidden).map { Float(($0 * 7) % 17) / 17 - 0.5 }
            let posX = (0 ..< gridW * hidden).map { Float(($0 * 3) % 11) / 11 }
            let posY = (0 ..< gridH * hidden).map { Float(($0 * 5) % 13) / 13 }

            var expected = [Float](repeating: 0, count: nPatches * hidden)
            for patch in 0 ..< nPatches {
                let col = patch % gridW
                let row = patch / gridW
                for d in 0 ..< hidden {
                    expected[patch * hidden + d] =
                        tokens[patch * hidden + d] + posX[col * hidden + d] + posY[row * hidden + d]
                }
            }
            let tokT = Tensor.empty(shape: [nPatches, hidden], dtype: .f32); tokT.copyIn(from: tokens)
            let pxT = Tensor.empty(shape: [gridW, hidden], dtype: .f32); pxT.copyIn(from: posX)
            let pyT = Tensor.empty(shape: [gridH, hidden], dtype: .f32); pyT.copyIn(from: posY)
            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.posEmb2dAdd(
                    tokens: tokT, posX: pxT, posY: pyT,
                    hidden: hidden, gridW: gridW, gridH: gridH, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            var maxDiff: Float = 0
            for i in 0 ..< expected.count { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-6, "posEmb2dAdd diverges (max \(maxDiff))")
        }
    }

    @Test("frameDiffLuma — GPU matches a CPU BT.601 luma SAD reference")
    func frameDiffLumaF32() {
        autoreleasepool {
            let (inH, inW, ds) = (8, 6, 2)
            let outH = inH / ds, outW = inW / ds
            let frame0 = (0 ..< inH * inW * 3).map { Float(($0 * 3 + 1) % 29) / 29 }
            let frame1 = (0 ..< inH * inW * 3).map { Float(($0 * 5 + 7) % 31) / 31 }
            let (wr, wg, wb): (Float, Float, Float) = (0.299, 0.587, 0.114)

            var expected = [Float](repeating: 0, count: outH * outW)
            let inv = 1 / Float(ds * ds)
            for oy in 0 ..< outH {
                for ox in 0 ..< outW {
                    var s0: Float = 0
                    var s1: Float = 0
                    for dy in 0 ..< ds {
                        for dx in 0 ..< ds {
                            let p = ((oy * ds + dy) * inW + (ox * ds + dx)) * 3
                            s0 += wr * frame0[p] + wg * frame0[p + 1] + wb * frame0[p + 2]
                            s1 += wr * frame1[p] + wg * frame1[p + 1] + wb * frame1[p + 2]
                        }
                    }
                    expected[oy * outW + ox] = abs(s0 * inv - s1 * inv)
                }
            }
            let f0 = Tensor.empty(shape: [inH, inW, 3], dtype: .f32); f0.copyIn(from: frame0)
            let f1 = Tensor.empty(shape: [inH, inW, 3], dtype: .f32); f1.copyIn(from: frame1)
            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.frameDiffLuma(frame0: f0, frame1: f1, inH: inH, inW: inW, ds: ds, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            #expect(outT.shape == [outH, outW])
            var maxDiff: Float = 0
            for i in 0 ..< expected.count { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-5, "frameDiffLuma diverges (max \(maxDiff))")
        }
    }

    @Test("adain1d — GPU matches a CPU instance-norm + affine reference")
    func adain1dF32() {
        autoreleasepool {
            let rows = 5, length = 7
            let eps: Float = 1e-5
            let x = (0 ..< rows * length).map { Float(($0 * 3 + 1) % 19) / 19 - 0.4 }
            let gamma = (0 ..< rows).map { Float($0 + 1) * 0.3 }
            let beta = (0 ..< rows).map { Float($0) * 0.1 - 0.2 }

            var expected = [Float](repeating: 0, count: rows * length)
            for r in 0 ..< rows {
                var s: Float = 0, sq: Float = 0
                for t in 0 ..< length { let v = x[r * length + t]; s += v; sq += v * v }
                let mean = s / Float(length)
                let varr = sq / Float(length) - mean * mean
                let inv = 1 / (varr + eps).squareRoot()
                for t in 0 ..< length {
                    expected[r * length + t] = (x[r * length + t] - mean) * inv * gamma[r] + beta[r]
                }
            }
            let xT = Tensor.empty(shape: [rows, length], dtype: .f32); xT.copyIn(from: x)
            let gT = Tensor.empty(shape: [rows], dtype: .f32); gT.copyIn(from: gamma)
            let bT = Tensor.empty(shape: [rows], dtype: .f32); bT.copyIn(from: beta)
            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.adain1d(xT, gamma: gT, beta: bT, rows: rows, length: length, eps: eps, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            var maxDiff: Float = 0
            for i in 0 ..< expected.count { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-4, "adain1d diverges (max \(maxDiff))")
        }
    }

    @Test("lstm — GPU full-sequence matches a CPU nn.LSTM reference (forward)")
    func lstmF32() {
        autoreleasepool {
            let seqLen = 5, inputDim = 3, hidden = 4
            func ramp(_ n: Int, _ seed: Int, _ amp: Float) -> [Float] {
                (0 ..< n).map { (Float(($0 * 7 + seed) % 11) / 11 - 0.5) * amp }
            }
            let x = ramp(seqLen * inputDim, 1, 1.0)
            let wIH = ramp(4 * hidden * inputDim, 2, 0.6)  // gate order i,f,g,o
            let wHH = ramp(4 * hidden * hidden, 3, 0.5)
            let bias = ramp(4 * hidden, 4, 0.3)

            func sig(_ v: Float) -> Float { 1 / (1 + Foundation.exp(-v)) }
            var h = [Float](repeating: 0, count: hidden)
            var c = [Float](repeating: 0, count: hidden)
            var expected = [Float](repeating: 0, count: seqLen * hidden)
            for t in 0 ..< seqLen {
                var g = [Float](repeating: 0, count: 4 * hidden)
                for r in 0 ..< 4 * hidden {
                    var acc = bias[r]
                    for d in 0 ..< inputDim { acc += wIH[r * inputDim + d] * x[t * inputDim + d] }
                    for d in 0 ..< hidden { acc += wHH[r * hidden + d] * h[d] }
                    g[r] = acc
                }
                var hn = [Float](repeating: 0, count: hidden)
                for j in 0 ..< hidden {
                    let ig = sig(g[0 * hidden + j])
                    let fg = sig(g[1 * hidden + j])
                    let gg = Foundation.tanh(g[2 * hidden + j])
                    let og = sig(g[3 * hidden + j])
                    c[j] = fg * c[j] + ig * gg
                    hn[j] = og * Foundation.tanh(c[j])
                }
                h = hn
                for j in 0 ..< hidden { expected[t * hidden + j] = h[j] }
            }
            let xT = Tensor.empty(shape: [seqLen, inputDim], dtype: .f32); xT.copyIn(from: x)
            let wihT = Tensor.empty(shape: [4 * hidden, inputDim], dtype: .f32); wihT.copyIn(from: wIH)
            let whhT = Tensor.empty(shape: [4 * hidden, hidden], dtype: .f32); whhT.copyIn(from: wHH)
            let bT = Tensor.empty(shape: [4 * hidden], dtype: .f32); bT.copyIn(from: bias)
            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.lstm(
                    x: xT, wIH: wihT, wHH: whhT, bias: bT,
                    seqLen: seqLen, inputDim: inputDim, hidden: hidden, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            var maxDiff: Float = 0
            for i in 0 ..< expected.count { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-4, "lstm diverges from nn.LSTM oracle (max \(maxDiff))")
        }
    }

    // Real Kokoro `predictor.shared` shape (hidden 256 = 8 simdgroups, long
    // sequence): exercises the cross-simdgroup threadgroup-memory hand-off the
    // short test (hidden 4, seqLen 5) never hits. Checks determinism across
    // repeated dispatches (a race would vary or NaN) + a CPU oracle.
    @Test("lstm — hidden 256 / seqLen 65 is finite, deterministic, matches CPU")
    func lstmLongSeqHidden256() {
        autoreleasepool {
            let seqLen = 65, inputDim = 640, hidden = 256
            func ramp(_ n: Int, _ seed: Int, _ amp: Float) -> [Float] {
                (0 ..< n).map { (Float(($0 * 13 + seed) % 23) / 23 - 0.5) * amp }
            }
            let x = ramp(seqLen * inputDim, 1, 2.0)
            let wIH = ramp(4 * hidden * inputDim, 2, 0.3)
            let wHH = ramp(4 * hidden * hidden, 3, 0.3)
            let bias = ramp(4 * hidden, 4, 0.2)
            func sig(_ v: Float) -> Float { 1 / (1 + Foundation.exp(-v)) }
            var h = [Float](repeating: 0, count: hidden), c = [Float](repeating: 0, count: hidden)
            var expected = [Float](repeating: 0, count: seqLen * hidden)
            for t in 0 ..< seqLen {
                var g = [Float](repeating: 0, count: 4 * hidden)
                for r in 0 ..< 4 * hidden {
                    var acc = bias[r]
                    for d in 0 ..< inputDim { acc += wIH[r * inputDim + d] * x[t * inputDim + d] }
                    for d in 0 ..< hidden { acc += wHH[r * hidden + d] * h[d] }
                    g[r] = acc
                }
                for j in 0 ..< hidden {
                    c[j] = sig(g[hidden + j]) * c[j] + sig(g[j]) * Foundation.tanh(g[2 * hidden + j])
                    h[j] = sig(g[3 * hidden + j]) * Foundation.tanh(c[j])
                }
                for j in 0 ..< hidden { expected[t * hidden + j] = h[j] }
            }
            let xT = Tensor.empty(shape: [seqLen, inputDim], dtype: .f32); xT.copyIn(from: x)
            let wihT = Tensor.empty(shape: [4 * hidden, inputDim], dtype: .f32); wihT.copyIn(from: wIH)
            let whhT = Tensor.empty(shape: [4 * hidden, hidden], dtype: .f32); whhT.copyIn(from: wHH)
            let bT = Tensor.empty(shape: [4 * hidden], dtype: .f32); bT.copyIn(from: bias)
            var first: [Float]? = nil
            for run in 0 ..< 12 {
                var outT: Tensor!
                runAndWait { cmd in
                    outT = Ops.lstm(x: xT, wIH: wihT, wHH: whhT, bias: bT,
                        seqLen: seqLen, inputDim: inputDim, hidden: hidden, on: cmd)
                }
                let got = outT.toArray(as: Float.self)
                #expect(got.allSatisfy { $0.isFinite }, "lstm run \(run): non-finite output")
                var maxDiff: Float = 0
                for i in 0 ..< expected.count { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
                #expect(maxDiff < 1e-3, "lstm run \(run): diverges from CPU oracle (max \(maxDiff))")
                if let f = first {
                    var rd: Float = 0
                    for i in 0 ..< got.count { rd = max(rd, abs(got[i] - f[i])) }
                    #expect(rd == 0, "lstm run \(run): non-deterministic vs run 0 (max \(rd)) — race")
                } else { first = got }
            }
        }
    }

    // The biLSTM shared-output pattern (two dispatches into one buffer via
    // outStride/outOffset, one reverse) at the real Kokoro shape — what
    // KokoroFrontEndGPU.biLstmGPU does. The plain-LSTM test above never
    // exercises the outOffset/reverse path or the shared `out` buffer.
    @Test("lstm — bidirectional shared-output (outOffset/reverse) is finite + deterministic")
    func lstmBidirectionalSharedOutput() {
        autoreleasepool {
            let seqLen = 65, inputDim = 640, hidden = 256
            func ramp(_ n: Int, _ s: Int, _ a: Float) -> [Float] {
                (0 ..< n).map { (Float(($0 * 13 + s) % 23) / 23 - 0.5) * a }
            }
            let x = ramp(seqLen * inputDim, 1, 2.0)
            let wihF = ramp(4 * hidden * inputDim, 2, 0.3), whhF = ramp(4 * hidden * hidden, 3, 0.3), bF = ramp(4 * hidden, 4, 0.2)
            let wihR = ramp(4 * hidden * inputDim, 5, 0.3), whhR = ramp(4 * hidden * hidden, 6, 0.3), bR = ramp(4 * hidden, 7, 0.2)
            let xT = Tensor.empty(shape: [seqLen, inputDim], dtype: .f32); xT.copyIn(from: x)
            func t2(_ a: [Float], _ s: [Int]) -> Tensor { let t = Tensor.empty(shape: s, dtype: .f32); t.copyIn(from: a); return t }
            let (wihFT, whhFT, bFT) = (t2(wihF, [4 * hidden, inputDim]), t2(whhF, [4 * hidden, hidden]), t2(bF, [4 * hidden]))
            let (wihRT, whhRT, bRT) = (t2(wihR, [4 * hidden, inputDim]), t2(whhR, [4 * hidden, hidden]), t2(bR, [4 * hidden]))
            var first: [Float]? = nil
            for run in 0 ..< 16 {
                let out = Tensor.empty(shape: [seqLen, 2 * hidden], dtype: .f32)
                runAndWait { cmd in
                    _ = Ops.lstm(x: xT, wIH: wihFT, wHH: whhFT, bias: bFT, seqLen: seqLen, inputDim: inputDim,
                        hidden: hidden, reverse: false, outStride: 2 * hidden, outOffset: 0, on: cmd, into: out)
                    _ = Ops.lstm(x: xT, wIH: wihRT, wHH: whhRT, bias: bRT, seqLen: seqLen, inputDim: inputDim,
                        hidden: hidden, reverse: true, outStride: 2 * hidden, outOffset: hidden, on: cmd, into: out)
                }
                let got = out.toArray(as: Float.self)
                #expect(got.allSatisfy { $0.isFinite }, "biLSTM run \(run): non-finite output")
                if let f = first {
                    var rd: Float = 0
                    for i in 0 ..< got.count { rd = max(rd, abs(got[i] - f[i])) }
                    #expect(rd == 0, "biLSTM run \(run): non-deterministic vs run 0 (max \(rd)) — race in the shared-output path")
                } else { first = got }
            }
        }
    }

    @Test("conv1dDilated — GPU matches a CPU dilated NCL conv reference (OIK)")
    func conv1dDilatedF32() {
        autoreleasepool {
            let (batch, inCh, inLen, outCh) = (1, 3, 12, 4)
            let (k, stride, pad, dilation) = (3, 1, 2, 2)  // 'same'-ish dilated
            let outLen = (inLen + 2 * pad - dilation * (k - 1) - 1) / stride + 1
            func ramp(_ n: Int, _ s: Int) -> [Float] { (0 ..< n).map { Float(($0 * 5 + s) % 13) / 13 - 0.4 } }
            let input = ramp(batch * inCh * inLen, 1)
            let weight = ramp(outCh * inCh * k, 2)  // OIK
            let bias = ramp(outCh, 3)

            var expected = [Float](repeating: 0, count: batch * outCh * outLen)
            for n in 0 ..< batch {
                for oc in 0 ..< outCh {
                    for op in 0 ..< outLen {
                        var acc = bias[oc]
                        for ic in 0 ..< inCh {
                            for kx in 0 ..< k {
                                let p = op * stride + kx * dilation
                                guard p >= pad && p < pad + inLen else { continue }
                                let ix = p - pad
                                acc += input[(n * inCh + ic) * inLen + ix] * weight[(oc * inCh + ic) * k + kx]
                            }
                        }
                        expected[(n * outCh + oc) * outLen + op] = acc
                    }
                }
            }
            let inT = Tensor.empty(shape: [batch, inCh, inLen], dtype: .f32); inT.copyIn(from: input)
            let wT = Tensor.empty(shape: [outCh, inCh, k], dtype: .f32); wT.copyIn(from: weight)
            let bT = Tensor.empty(shape: [outCh], dtype: .f32); bT.copyIn(from: bias)
            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.conv1dDilated(
                    inT, weight: wT, bias: bT, batch: batch, inCh: inCh, inLen: inLen,
                    outCh: outCh, k: k, stride: stride, pad: pad, dilation: dilation, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            #expect(outT.shape == [batch, outCh, outLen])
            var maxDiff: Float = 0
            for i in 0 ..< expected.count { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-4, "conv1dDilated diverges (max \(maxDiff))")
        }
    }

    @Test("conv1dTranspose — GPU matches a CPU ConvTranspose1d NCL reference (IOK)")
    func conv1dTransposeF32() {
        autoreleasepool {
            let (batch, inCh, inLen, outCh) = (1, 3, 6, 2)
            let (k, stride, pad, dilation, outPad) = (4, 2, 1, 1, 0)
            let outLen = (inLen - 1) * stride - 2 * pad + dilation * (k - 1) + outPad + 1
            func ramp(_ n: Int, _ s: Int) -> [Float] { (0 ..< n).map { Float(($0 * 7 + s) % 17) / 17 - 0.4 } }
            let input = ramp(batch * inCh * inLen, 1)
            let weight = ramp(inCh * outCh * k, 2)  // IOK
            let bias = ramp(outCh, 3)

            var expected = [Float](repeating: 0, count: batch * outCh * outLen)
            for n in 0 ..< batch {
                for oc in 0 ..< outCh {
                    for op in 0 ..< outLen {
                        var acc = bias[oc]
                        let opp = op + pad
                        for ic in 0 ..< inCh {
                            for kx in 0 ..< k {
                                let tap = kx * dilation
                                guard opp >= tap else { continue }
                                let num = opp - tap
                                guard num % stride == 0 else { continue }
                                let ip = num / stride
                                guard ip < inLen else { continue }
                                acc +=
                                    input[(n * inCh + ic) * inLen + ip]
                                    * weight[(ic * outCh + oc) * k + kx]
                            }
                        }
                        expected[(n * outCh + oc) * outLen + op] = acc
                    }
                }
            }
            let inT = Tensor.empty(shape: [batch, inCh, inLen], dtype: .f32); inT.copyIn(from: input)
            let wT = Tensor.empty(shape: [inCh, outCh, k], dtype: .f32); wT.copyIn(from: weight)
            let bT = Tensor.empty(shape: [outCh], dtype: .f32); bT.copyIn(from: bias)
            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.conv1dTranspose(
                    inT, weight: wT, bias: bT, batch: batch, inCh: inCh, inLen: inLen,
                    outCh: outCh, outLen: outLen, k: k, stride: stride, pad: pad, dilation: dilation, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            #expect(outT.shape == [batch, outCh, outLen])
            var maxDiff: Float = 0
            for i in 0 ..< expected.count { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-4, "conv1dTranspose diverges (max \(maxDiff))")
        }
    }

    @Test("leakyReLU — GPU matches a CPU leaky rectifier (slope 0.2)")
    func leakyReLUF32() {
        autoreleasepool {
            let n = 257
            let slope: Float = 0.2
            let x = (0 ..< n).map { Float($0 - 128) * 0.1 }
            let expected = x.map { $0 > 0 ? $0 : slope * $0 }
            let xT = Tensor.empty(shape: [n], dtype: .f32); xT.copyIn(from: x)
            var outT: Tensor!
            runAndWait { cmd in outT = Ops.leakyReLU(xT, slope: slope, on: cmd) }
            let got = outT.toArray(as: Float.self)
            var maxDiff: Float = 0
            for i in 0 ..< n { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-5, "leakyReLU diverges (max \(maxDiff))")
        }
    }

    @Test("snake — GPU matches a CPU snake reference (per-channel alpha)")
    func snakeF32() {
        autoreleasepool {
            let channels = 5, length = 37
            let alpha = (0 ..< channels).map { Float(0.5) + 0.3 * Float($0) }
            let x = (0 ..< channels * length).map { (Float($0 % 19) / 19 - 0.5) * 3 }
            var expected = [Float](repeating: 0, count: channels * length)
            for i in 0 ..< channels * length {
                let a = alpha[i / length]
                let s = sin(a * x[i])
                expected[i] = x[i] + (1 / (a + 1e-9)) * s * s
            }
            let xT = Tensor.empty(shape: [channels, length], dtype: .f32); xT.copyIn(from: x)
            let aT = Tensor.empty(shape: [channels], dtype: .f32); aT.copyIn(from: alpha)
            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.snake(xT, alpha: aT, channels: channels, length: length, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            var maxDiff: Float = 0
            for i in 0 ..< channels * length { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-4, "snake diverges (max \(maxDiff))")
        }
    }

    @Test("upsampleNearest1d — GPU matches a CPU nearest ×2 reference")
    func upsampleNearest1dF32() {
        autoreleasepool {
            let channels = 6, inLen = 29, factor = 2
            let outLen = inLen * factor
            let x = (0 ..< channels * inLen).map { Float($0 % 23) / 23 - 0.5 }
            var expected = [Float](repeating: 0, count: channels * outLen)
            for i in 0 ..< channels * outLen {
                let c = i / outLen, t = (i % outLen) / factor
                expected[i] = x[c * inLen + t]
            }
            let xT = Tensor.empty(shape: [channels, inLen], dtype: .f32); xT.copyIn(from: x)
            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.upsampleNearest1d(xT, channels: channels, inLen: inLen, factor: factor, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            #expect(outT.shape == [channels, outLen])
            var maxDiff: Float = 0
            for i in 0 ..< channels * outLen { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-5, "upsampleNearest1d diverges (max \(maxDiff))")
        }
    }

    @Test("geluErf — GPU matches an exact erf-GELU reference")
    func geluErfF32() {
        autoreleasepool {
            let n = 256
            let x = (0 ..< n).map { Float($0 - 128) * 0.05 }
            let expected = x.map { 0.5 * $0 * (1 + erff($0 * 0.70710678118654752)) }
            let xT = Tensor.empty(shape: [n], dtype: .f32); xT.copyIn(from: x)
            var outT: Tensor!
            runAndWait { cmd in outT = Ops.geluErf(xT, on: cmd) }
            let got = outT.toArray(as: Float.self)
            var maxDiff: Float = 0
            for i in 0 ..< n { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-5, "geluErf diverges (max \(maxDiff))")
        }
    }

    @Test("conv1dTransposeDepthwise — GPU matches a CPU depthwise convT (k3 s2 pad1)")
    func conv1dTransposeDepthwiseF32() {
        autoreleasepool {
            let (channels, inLen, k, stride, pad, dilation) = (6, 13, 3, 2, 1, 1)
            let outLen = (inLen - 1) * stride - 2 * pad + dilation * (k - 1) + 1
            func ramp(_ n: Int, _ s: Int, _ amp: Float) -> [Float] { (0 ..< n).map { (Float(($0 * 7 + s) % 17) / 17 - 0.5) * amp } }
            let input = ramp(channels * inLen, 1, 4)
            let weight = ramp(channels * k, 2, 2)
            let bias = ramp(channels, 3, 0.5)
            var expected = [Float](repeating: 0, count: channels * outLen)
            for c in 0 ..< channels {
                for op in 0 ..< outLen {
                    var acc = bias[c]
                    let opp = op + pad
                    for kx in 0 ..< k {
                        let tap = kx * dilation
                        guard opp >= tap else { continue }
                        let num = opp - tap
                        guard num % stride == 0 else { continue }
                        let ip = num / stride
                        guard ip < inLen else { continue }
                        acc += input[c * inLen + ip] * weight[c * k + kx]
                    }
                    expected[c * outLen + op] = acc
                }
            }
            let inT = Tensor.empty(shape: [channels, inLen], dtype: .f32); inT.copyIn(from: input)
            let wT = Tensor.empty(shape: [channels, k], dtype: .f32); wT.copyIn(from: weight)
            let bT = Tensor.empty(shape: [channels], dtype: .f32); bT.copyIn(from: bias)
            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.conv1dTransposeDepthwise(
                    inT, weight: wT, bias: bT, channels: channels, inLen: inLen,
                    k: k, stride: stride, pad: pad, dilation: dilation, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            #expect(outT.shape == [channels, outLen])
            var maxDiff: Float = 0
            for i in 0 ..< channels * outLen { maxDiff = max(maxDiff, abs(got[i] - expected[i])) }
            #expect(maxDiff < 1e-4, "conv1dTransposeDepthwise diverges (max \(maxDiff))")
        }
    }

    @Test("depthwiseConv2dNHWC — GPU matches a CPU NHWC reference (3×3 s1 pad1)")
    func depthwiseConv2dNHWC() {
        autoreleasepool {
            let batch = 1, ch = 4, inH = 8, inW = 8
            let kH = 3, kW = 3, stride = 1, pad = 1, dilation = 1
            let outH = (inH + 2 * pad - dilation * (kH - 1) - 1) / stride + 1
            let outW = (inW + 2 * pad - dilation * (kW - 1) - 1) / stride + 1

            func ramp(_ n: Int, _ period: Int, _ amp: Float) -> [Float] {
                (0 ..< n).map { (Float($0 % period) / Float(period) - 0.5) * amp }
            }
            let input = ramp(batch * inH * inW * ch, 13, 6.0)
            let weight = ramp(ch * kH * kW, 11, 4.0)
            let bias = ramp(ch, 5, 2.0)

            // CPU NHWC reference.
            var expected = [Float](repeating: 0, count: batch * outH * outW * ch)
            for n in 0 ..< batch {
                for oh in 0 ..< outH {
                    for ow in 0 ..< outW {
                        for c in 0 ..< ch {
                            var acc = bias[c]
                            for ky in 0 ..< kH {
                                let ph = oh * stride + ky * dilation
                                if ph < pad || ph >= pad + inH { continue }
                                let ih = ph - pad
                                for kx in 0 ..< kW {
                                    let pw = ow * stride + kx * dilation
                                    if pw < pad || pw >= pad + inW { continue }
                                    let iw = pw - pad
                                    let inIdx = ((n * inH + ih) * inW + iw) * ch + c
                                    let wIdx = (c * kH + ky) * kW + kx
                                    acc += input[inIdx] * weight[wIdx]
                                }
                            }
                            expected[((n * outH + oh) * outW + ow) * ch + c] = acc
                        }
                    }
                }
            }

            let inT = Tensor.empty(shape: [batch, inH, inW, ch], dtype: .f32)
            inT.copyIn(from: input)
            let wT = Tensor.empty(shape: [ch, kH, kW], dtype: .f32)
            wT.copyIn(from: weight)
            let bT = Tensor.empty(shape: [ch], dtype: .f32)
            bT.copyIn(from: bias)

            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.depthwiseConv2dNHWC(
                    input: inT, weight: wT, bias: bT,
                    batch: batch, channels: ch, inH: inH, inW: inW,
                    kH: kH, kW: kW, stride: stride, padding: pad, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            #expect(got.count == expected.count)
            var maxDiff: Float = 0
            for i in 0 ..< min(got.count, expected.count) {
                maxDiff = max(maxDiff, abs(got[i] - expected[i]))
            }
            #expect(
                maxDiff < 1e-4,
                "depthwiseConv2dNHWC GPU diverges from CPU reference (max |Δ| \(maxDiff))")
        }
    }

    @Test("sdpaBidirectionalWindowed — GPU matches a CPU windowed reference (d80)")
    func sdpaBidirectionalWindowed() {
        autoreleasepool {
            // MHA, head_dim 80, ragged windows (last short) to exercise the
            // non-uniform case Qwen2.5-VL's native-resolution grid produces.
            let headDim = 80
            let nQHeads = 4, nKVHeads = 4
            let nQuery = 20, win = 8
            let kvStride = nQuery
            let scale = 1.0 / Float(Double(headDim).squareRoot())
            let gqa = nQHeads / nKVHeads

            // Window partition: [0,8),[8,16),[16,20).
            var segStart = [UInt32](repeating: 0, count: nQuery)
            var segLen = [UInt32](repeating: 0, count: nQuery)
            for i in 0 ..< nQuery {
                let ws = (i / win) * win
                let we = min(ws + win, nQuery)
                segStart[i] = UInt32(ws)
                segLen[i] = UInt32(we - ws)
            }

            func ramp(_ n: Int, _ step: Float, _ start: Float) -> [Float] {
                (0 ..< n).map { (start + Float($0) * step).truncatingRemainder(dividingBy: 2.0) - 1.0 }
            }
            let q = ramp(nQuery * nQHeads * headDim, 0.013, -0.4)
            let k = ramp(nKVHeads * kvStride * headDim, 0.011, -0.5)
            let v = ramp(nKVHeads * kvStride * headDim, 0.007, -0.3)

            // CPU windowed reference: each query softmaxes over its window.
            var expected = [Float](repeating: 0, count: nQuery * nQHeads * headDim)
            for r in 0 ..< nQuery {
                let ws = Int(segStart[r])
                let we = ws + Int(segLen[r])
                for qh in 0 ..< nQHeads {
                    let kvh = qh / gqa
                    let qOff = (r * nQHeads + qh) * headDim
                    let kvSlab = kvh * kvStride * headDim
                    var scores = [Float](repeating: 0, count: we - ws)
                    for (idx, t) in (ws ..< we).enumerated() {
                        let kOff = kvSlab + t * headDim
                        var dot: Float = 0
                        for d in 0 ..< headDim { dot += q[qOff + d] * k[kOff + d] }
                        scores[idx] = dot * scale
                    }
                    let m = scores.max() ?? 0
                    var sum: Float = 0
                    for i in scores.indices { scores[i] = exp(scores[i] - m); sum += scores[i] }
                    let inv = sum > 0 ? 1 / sum : 0
                    for d in 0 ..< headDim {
                        var acc: Float = 0
                        for (idx, t) in (ws ..< we).enumerated() {
                            acc += scores[idx] * inv * v[kvSlab + t * headDim + d]
                        }
                        expected[qOff + d] = acc
                    }
                }
            }

            let qT = Tensor.empty(shape: [nQuery, nQHeads, headDim], dtype: .f32)
            qT.copyIn(from: q)
            let kT = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            kT.copyIn(from: k)
            let vT = Tensor.empty(shape: [nKVHeads, kvStride, headDim], dtype: .f32)
            vT.copyIn(from: v)
            let ssT = Tensor.empty(shape: [nQuery], dtype: .u32)
            ssT.copyIn(from: segStart)
            let slT = Tensor.empty(shape: [nQuery], dtype: .u32)
            slT.copyIn(from: segLen)

            var outT: Tensor!
            runAndWait { cmd in
                outT = Ops.sdpaBidirectionalWindowed(
                    q: qT, k: kT, v: vT, segStart: ssT, segLen: slT,
                    nQHeads: nQHeads, nKVHeads: nKVHeads,
                    nQuery: nQuery, kvStride: kvStride, scale: scale, on: cmd)
            }
            let got = outT.toArray(as: Float.self)
            #expect(got.count == expected.count)
            var maxDiff: Float = 0
            for i in 0 ..< min(got.count, expected.count) {
                maxDiff = max(maxDiff, abs(got[i] - expected[i]))
            }
            #expect(
                maxDiff < 1e-3,
                "sdpaBidirectionalWindowed GPU diverges from CPU reference (max |Δ| \(maxDiff))")
        }
    }
}

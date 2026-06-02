import Metal
import Foundation

// Standalone correctness+perf harness for the hand-tuned MoE IQ2_XXS GEMM.
// Single expert across all rows (correctness focus). Compares GPU vs a CPU
// dequant+matmul reference, then benches.

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let src = try String(contentsOfFile: "/tmp/moe_mma_dev/kernel.metal", encoding: .utf8)
let lib = try dev.makeLibrary(source: src, options: nil)
let pso = try dev.makeComputePipelineState(function: lib.makeFunction(name: "moe_mma_iq2")!)

let M = Int(ProcessInfo.processInfo.environment["M"] ?? "64")!, K = 4096, N = 2048
let doRef = M <= 512
let nblk = N * K / 256            // blocks for the whole expert tensor [N,K]
let blkU16 = 33                  // u16 per block

// Random IQ2 data (u16). d (u16[0] of each block) = small f16.
func f16bits(_ x: Float) -> UInt16 { Float16(x).bitPattern }
var rng: UInt64 = 0xBEEF
func r16() -> UInt16 { rng = rng &* 6364136223846793005 &+ 1; return UInt16((rng >> 33) & 0xffff) }
var view = [UInt16](repeating: 0, count: nblk * blkU16)
for b in 0..<nblk {
    view[b*blkU16] = f16bits((Float(Int(r16() % 200)) - 100) * 0.001)
    for i in 1..<blkU16 { view[b*blkU16 + i] = r16() }
}
let gridT: [UInt8] = (0..<2048).map { UInt8(($0 * 7) % 256) }
let signsT: [UInt8] = (0..<128).map { UInt8(($0 * 13) % 256) }
var x = [Float16](repeating: 0, count: M * K)
for i in 0..<M*K { x[i] = Float16((Float((i % 17)) - 8) * 0.01) }

// CPU reference: dequant W[n,k] then out[m,n] = sum_k x[m,k]*W[n,k].
func dequantW(_ n: Int, _ k: Int) -> Float {
    let vidx = n * K + k
    let block = vidx / 256, group = (vidx & 255) / 32
    let b16 = block * blkU16
    let d = Float(Float16(bitPattern: view[b16]))
    let q0 = b16 + 1 + group * 4
    let auxIdx = UInt32(view[q0]) | (UInt32(view[q0+1]) << 16)
    let auxSgn = UInt32(view[q0+2]) | (UInt32(view[q0+3]) << 16)
    let scale4 = auxSgn >> 28
    let db = d * ((Float(scale4) + 0.5) * 0.25)
    let kl = k & 255
    let oct = (UInt32(kl) & 31) / 8, lo = UInt32(kl) & 7
    let gkey = (auxIdx >> (oct * 8)) & 0xff
    let o = Float(Int8(bitPattern: gridT[Int(gkey) * 8 + Int(lo)]))
    let sidx = (auxSgn >> (oct * 7)) & 0x7f
    let sign: Float = ((UInt32(signsT[Int(sidx)]) >> lo) & 1) != 0 ? -1 : 1
    return db * sign * o
}
var ref = [Float](repeating: 0, count: doRef ? M * N : 0)
if doRef {
    print("computing CPU reference (\(M)x\(N))...")
    DispatchQueue.concurrentPerform(iterations: N) { n in
        var wcol = [Float](repeating: 0, count: K)
        for k in 0..<K { wcol[k] = dequantW(n, k) }
        for m in 0..<M {
            var acc: Float = 0
            for k in 0..<K { acc += Float(x[m*K+k]) * wcol[k] }
            ref[m*N + n] = acc
        }
    }
}

func buf<T>(_ a: [T]) -> MTLBuffer { dev.makeBuffer(bytes: a, length: MemoryLayout<T>.stride*a.count)! }
let xB = buf(x), vB = buf(view), gB = buf(gridT), sB = buf(signsT)
let idxB = buf([UInt32](repeating: 0, count: M))
let outB = dev.makeBuffer(length: M*N*2)!
var mt = UInt32(M), no = UInt32(N), ki = UInt32(K), tbo: UInt32 = 0, ebs = UInt32(nblk*66)

func run() -> Double {
    memset(outB.contents(), 0, M*N*2)
    let cb = q.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(xB,offset:0,index:0); enc.setBuffer(vB,offset:0,index:1); enc.setBuffer(vB,offset:0,index:2)
    enc.setBuffer(gB,offset:0,index:3); enc.setBuffer(sB,offset:0,index:4); enc.setBuffer(idxB,offset:0,index:5); enc.setBuffer(outB,offset:0,index:6)
    enc.setBytes(&mt,length:4,index:7); enc.setBytes(&no,length:4,index:8); enc.setBytes(&ki,length:4,index:9); enc.setBytes(&tbo,length:4,index:10); enc.setBytes(&ebs,length:4,index:11)
    enc.dispatchThreadgroups(MTLSize(width: N/64, height: (M+63)/64, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
    let t0 = Date(); cb.commit(); cb.waitUntilCompleted()
    return Date().timeIntervalSince(t0) * 1000
}
_ = run()
let outP = outB.contents().bindMemory(to: Float16.self, capacity: M*N)
if doRef {
    var worst: Float = 0, wm = 0, wn = 0
    for m in 0..<M { for n in 0..<N {
        let g = Float(outP[m*N+n]); let d = abs(g - ref[m*N+n])
        if d > worst { worst = d; wm = m; wn = n }
    }}
    let mag = ref.map { abs($0) }.max() ?? 1
    print(String(format: "CORRECTNESS worst=%.4e @ (%d,%d) gpu=%.4f ref=%.4f mag=%.3f -> %@",
        worst, wm, wn, Float(outP[wm*N+wn]), ref[wm*N+wn], mag, worst < mag*5e-2 ? "PASS" : "FAIL"))
}
var best = Double.greatestFiniteMagnitude
for _ in 0..<20 { best = min(best, run()) }
print(String(format: "PERF M=%d best=%.3f ms", M, best))

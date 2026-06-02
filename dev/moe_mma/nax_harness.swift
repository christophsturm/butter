import Metal
import Foundation
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let src = try String(contentsOfFile: "/tmp/moe_mma_dev/nax_probe.metal", encoding: .utf8)
let lib = try dev.makeLibrary(source: src, options: nil)
let M = Int(ProcessInfo.processInfo.environment["M"] ?? "4096")!
let N = Int(ProcessInfo.processInfo.environment["N"] ?? "4096")!
let K = Int(ProcessInfo.processInfo.environment["K"] ?? "4096")!
var rng: UInt64 = 1
func r() -> Float16 { rng = rng &* 6364136223846793005 &+ 1; return Float16((Float(Int(rng>>40)&0xff)-128)*0.01) }
let A = (0..<M*K).map { _ in r() }, B = (0..<N*K).map { _ in r() }
func buf<T>(_ a:[T])->MTLBuffer{dev.makeBuffer(bytes:a,length:MemoryLayout<T>.stride*a.count)!}
let aB=buf(A), bB=buf(B)
var m=UInt32(M),n=UInt32(N),k=UInt32(K)
func runK(_ name:String)->(Double,[Float16])? {
    guard let fn = lib.makeFunction(name:name) else { print("\(name): MISSING"); return nil }
    let pso = try! dev.makeComputePipelineState(function:fn)
    let cB = dev.makeBuffer(length:M*N*2)!
    func once()->Double{
        memset(cB.contents(),0,M*N*2)
        let cb=q.makeCommandBuffer()!, e=cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pso); e.setBuffer(aB,offset:0,index:0); e.setBuffer(bB,offset:0,index:1); e.setBuffer(cB,offset:0,index:2)
        e.setBytes(&m,length:4,index:3); e.setBytes(&n,length:4,index:4); e.setBytes(&k,length:4,index:5)
        e.dispatchThreadgroups(MTLSize(width:N/64,height:(M+63)/64,depth:1),threadsPerThreadgroup:MTLSize(width:128,height:1,depth:1))
        e.endEncoding(); let t=Date(); cb.commit(); cb.waitUntilCompleted(); return Date().timeIntervalSince(t)*1000
    }
    _=once(); var best=Double.greatestFiniteMagnitude; for _ in 0..<30 { best=min(best,once()) }
    let out=cB.contents().bindMemory(to:Float16.self,capacity:M*N); return (best,(0..<M*N).map{out[$0]})
}
let sg = runK("mm_sg"); let mpp = runK("mm_mpp")
if let (ts,cs)=sg { print(String(format:"mm_sg   %dx%dx%d: %.3f ms",M,N,K,ts))
  if let (tm,cm)=mpp { print(String(format:"mm_mpp  %dx%dx%d: %.3f ms  (matmul2d/NAX)",M,N,K,tm))
    var w:Float=0; for i in 0..<min(cs.count,cm.count){ let d=abs(Float(cs[i])-Float(cm[i])); if d>w{w=d} }
    print(String(format:"  outputs max|Δ|=%.3f (should be ~0 if same GEMM)  speedup sg/mpp=%.2fx",w,ts/tm))
  } }

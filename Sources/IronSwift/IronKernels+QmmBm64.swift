import Metal

extension IronKernels {
    /// Overlay dispatch for `iron_qmm_coop_bm64_bf16` in kernels_extra.metallib.
    /// Grid is threadgroups: `[n/128, m/64, 1]`, tg `[256, 1, 1]`.
    public static func iron_qmm_coop_bm64_bf16_threadgroups(
        w: MTLBuffer, wOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        k: UInt32,
        n: UInt32,
        gs_per_row: UInt32,
        gridSize: MTLSize,
        threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) {
        let pso = PSOCache.shared.pipelineState(for: "iron_qmm_coop_bm64_bf16")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(w, offset: wOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(out, offset: outOffset, index: 4)
        var k_v = k
        enc.setBytes(&k_v, length: 4, index: 5)
        var n_v = n
        enc.setBytes(&n_v, length: 4, index: 6)
        var gs_per_row_v = gs_per_row
        enc.setBytes(&gs_per_row_v, length: 4, index: 7)
        enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        enc.endEncoding()
    }
}

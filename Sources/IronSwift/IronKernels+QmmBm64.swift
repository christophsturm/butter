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

    /// Overlay dispatch for a baked-K/N BM64 tile. Buffers 0-4 only.
    /// Grid is threadgroups: `[n/128, m/64, 1]`, tg `[256, 1, 1]`.
    private static func dispatchBm64Shape(
        name: String,
        w: MTLBuffer, wOffset: Int,
        scales: MTLBuffer, scalesOffset: Int,
        biases: MTLBuffer, biasesOffset: Int,
        x: MTLBuffer, xOffset: Int,
        out: MTLBuffer, outOffset: Int,
        gridSize: MTLSize,
        threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) {
        let pso = PSOCache.shared.pipelineState(for: name)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(w, offset: wOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(out, offset: outOffset, index: 4)
        enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        enc.endEncoding()
    }

    public static func iron_qmm_coop_bm64_shape_n17408_k5120_bf16_threadgroups(
        w: MTLBuffer, wOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        gridSize: MTLSize,
        threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) {
        dispatchBm64Shape(
            name: "iron_qmm_coop_bm64_shape_n17408_k5120",
            w: w, wOffset: wOffset, scales: scales, scalesOffset: scalesOffset,
            biases: biases, biasesOffset: biasesOffset, x: x, xOffset: xOffset,
            out: out, outOffset: outOffset, gridSize: gridSize,
            threadgroupSize: threadgroupSize, on: commandBuffer)
    }

    public static func iron_qmm_coop_bm64_shape_n10240_k5120_bf16_threadgroups(
        w: MTLBuffer, wOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        gridSize: MTLSize,
        threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) {
        dispatchBm64Shape(
            name: "iron_qmm_coop_bm64_shape_n10240_k5120",
            w: w, wOffset: wOffset, scales: scales, scalesOffset: scalesOffset,
            biases: biases, biasesOffset: biasesOffset, x: x, xOffset: xOffset,
            out: out, outOffset: outOffset, gridSize: gridSize,
            threadgroupSize: threadgroupSize, on: commandBuffer)
    }

    public static func iron_qmm_coop_bm64_shape_n5120_k17408_bf16_threadgroups(
        w: MTLBuffer, wOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        gridSize: MTLSize,
        threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) {
        dispatchBm64Shape(
            name: "iron_qmm_coop_bm64_shape_n5120_k17408",
            w: w, wOffset: wOffset, scales: scales, scalesOffset: scalesOffset,
            biases: biases, biasesOffset: biasesOffset, x: x, xOffset: xOffset,
            out: out, outOffset: outOffset, gridSize: gridSize,
            threadgroupSize: threadgroupSize, on: commandBuffer)
    }

    public static func iron_qmm_coop_bm64_shape_n6144_k5120_bf16_threadgroups(
        w: MTLBuffer, wOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        gridSize: MTLSize,
        threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) {
        dispatchBm64Shape(
            name: "iron_qmm_coop_bm64_shape_n6144_k5120",
            w: w, wOffset: wOffset, scales: scales, scalesOffset: scalesOffset,
            biases: biases, biasesOffset: biasesOffset, x: x, xOffset: xOffset,
            out: out, outOffset: outOffset, gridSize: gridSize,
            threadgroupSize: threadgroupSize, on: commandBuffer)
    }

    public static func iron_qmm_coop_bm64_shape_n5120_k6144_bf16_threadgroups(
        w: MTLBuffer, wOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        gridSize: MTLSize,
        threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) {
        dispatchBm64Shape(
            name: "iron_qmm_coop_bm64_shape_n5120_k6144",
            w: w, wOffset: wOffset, scales: scales, scalesOffset: scalesOffset,
            biases: biases, biasesOffset: biasesOffset, x: x, xOffset: xOffset,
            out: out, outOffset: outOffset, gridSize: gridSize,
            threadgroupSize: threadgroupSize, on: commandBuffer)
    }

    public static func iron_qmm_coop_bm64_shape_n1024_k5120_bf16_threadgroups(
        w: MTLBuffer, wOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        gridSize: MTLSize,
        threadgroupSize: MTLSize,
        on commandBuffer: MTLCommandBuffer
    ) {
        dispatchBm64Shape(
            name: "iron_qmm_coop_bm64_shape_n1024_k5120",
            w: w, wOffset: wOffset, scales: scales, scalesOffset: scalesOffset,
            biases: biases, biasesOffset: biasesOffset, x: x, xOffset: xOffset,
            out: out, outOffset: outOffset, gridSize: gridSize,
            threadgroupSize: threadgroupSize, on: commandBuffer)
    }
}

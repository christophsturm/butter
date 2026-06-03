// Copyright 2026 Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// `ffai dsv4bench` — standalone release-mode decode benchmark for the
// DeepSeek V4 Flash GGUF path. Exists because the swift-testing async
// harness segfaults in release when this model loads, so a plain CLI
// command is the way to get clean release TPS numbers.

import ArgumentParser
import FFAI
import Foundation

struct Dsv4BenchCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "dsv4bench",
        abstract: "Sustained decode benchmark for the DSv4-Flash GGUF model."
    )

    @Option(name: .shortAndLong, help: "Directory containing the DSv4 GGUF (or the .gguf file itself).")
    var model: String = NSString("~/models/deepseek-v4-flash").expandingTildeInPath

    @Option(name: .shortAndLong, help: "Number of decode tokens to time.")
    var tokens: Int = 8

    @Flag(
        name: .long,
        help:
            "Feed a fixed input token each step (no argmax feedback) — deterministic routing for clean steady-state perf comparison."
    )
    var fixed: Bool = false

    @Option(
        name: .long,
        help:
            "Simulate decoding deep in a context: pre-fill the sliding-window KV (full 128) and set RoPE position to this value (e.g. 32000)."
    )
    var startPos: Int = 0

    @Option(
        name: .long,
        help:
            "Literal prefill: run N sequential forwards over a varied token stream (exercises real expert routing) and report prefill tok/s, before timing decode."
    )
    var prefill: Int = 0

    @Option(
        name: .long,
        help:
            "Validate + time batched prefill: compare forwardPrefillChunk's last-token logits vs the sequential path on an N-token prompt, then report prefill tok/s."
    )
    var validatePrefill: Int = 0

    @Option(
        name: .long,
        help:
            "Comma-separated explicit token IDs: run forwardPrefillChunk on them and print the next-token argmax + top-8 logits (oracle comparison on the same IDs)."
    )
    var promptIds: String = ""

    func run() throws {
        // Run the whole bench on a dedicated thread rather than the
        // swift-argument-parser cooperative async executor, which the
        // DSv4 forward path does not run cleanly on in release builds.
        var caught: Error?
        let t = Thread {
            do { try self.runBody() } catch { caught = error }
        }
        t.stackSize = 64 << 20
        t.start()
        while !t.isFinished { usleep(2000) }
        if let caught { throw caught }
    }

    private func dbg(_ s: String) { FileHandle.standardError.write(Data(("[dsv4bench] " + s + "\n").utf8)) }

    private func runBody() throws {
        dbg("start runBody")
        let path = (model as NSString).expandingTildeInPath
        var isDir: ObjCBool = false
        FileManager.default.fileExists(atPath: path, isDirectory: &isDir)
        let bundle =
            isDir.boolValue
            ? try GGUFTensorBundle(directory: URL(fileURLWithPath: path))
            : try GGUFTensorBundle(url: URL(fileURLWithPath: path))
        dbg("bundle opened")

        let device = Device.shared
        if ProcessInfo.processInfo.environment["FFAI_VERIFY_VIEWS"] == "1" {
            // Verify the zero-copy mmap views return identical bytes to the
            // mmap read path, for a representative expert tensor.
            for name in ["blk.0.ffn_down_exps.weight", "blk.20.ffn_gate_exps.weight", "blk.42.ffn_up_exps.weight"] {
                guard let v = bundle.gpuTensorView(named: name, device: device) else {
                    dbg("VIEW \(name): nil"); continue
                }
                let viewBytes = v.buffer.contents().advanced(by: v.offset).assumingMemoryBound(to: UInt8.self)
                let ref = try bundle.reader.withRawBytes(named: name) { ptr -> [UInt8] in
                    Array(UnsafeBufferPointer(start: ptr.baseAddress!, count: 32))
                }
                var match = true
                for i in 0 ..< 32 where viewBytes[i] != ref[i] { match = false }
                dbg("VIEW \(name): off=\(v.offset) first32match=\(match)")
                // STRIDE CHECK (pure metadata): the view kernel addresses
                // expert E at off + E*(nblk*blockBytes). If that != the real
                // per-expert byte distance (byteLength/nExperts), experts>0
                // read wrong data while expert 0 stays correct — exactly the
                // observed gateP divergence.
                let idx = bundle.reader.tensorIndex[name]!
                let tinfo = bundle.reader.tensorInfos[idx]
                let nExp = 256
                let blockBytes = name.contains("down") ? 84 : 66
                let nblk = (Int(tinfo.numElements) / nExp) / 256
                let strideKernel = nblk * blockBytes
                let strideReal = Int(tinfo.byteLength) / nExp
                dbg(
                    "STRIDE \(name): numElem=\(tinfo.numElements) byteLen=\(tinfo.byteLength) dims=\(tinfo.dimensions) nblk=\(nblk) kernelStride=\(strideKernel) realStride=\(strideReal) MATCH=\(strideKernel == strideReal)"
                )
                // Byte-compare expert 5 read at each stride vs the raw mmap.
                let e = 5
                let rawE = try bundle.reader.withRawBytesSlice(named: name, byteStart: strideReal * e, byteLength: 32) {
                    Array(UnsafeBufferPointer(start: $0.baseAddress!, count: 32))
                }
                var matchK = true; var matchR = true
                for i in 0 ..< 32 {
                    if viewBytes[strideKernel * e + i] != rawE[i] { matchK = false }
                    if viewBytes[strideReal * e + i] != rawE[i] { matchR = false }
                }
                dbg("EXPERT5 \(name): viewAtKernelStride==raw? \(matchK)  viewAtRealStride==raw? \(matchR)")
            }
            return
        }
        dbg("device ready, loading model")
        let model = try DeepSeekV4Flash.loadFlashFromGGUF(
            gguf: bundle, device: device)
        dbg("model loaded")
        model.keepLayersResident = true
        let state = model.makeDecodeState()
        if startPos > 0 {
            // Simulate being deep in a long context: the sliding-window
            // KV is already full (nVisible caps at nSWA), RoPE at startPos.
            state.position = startPos
            for ls in state.layerStates { ls.swCount = ls.nSWA }
            dbg("seeded context: position=\(startPos), KV window full (\(state.layerStates.first?.nSWA ?? 0))")
        }
        dbg("state ready, starting decode")
        let bos = Int(bundle.reader.metadataUInt32("tokenizer.ggml.bos_token_id") ?? 0)
        let vocab = 129_280

        // ── Literal prefill: N sequential forwards over a varied token
        // stream (no parallel-prefill path exists; this is the only way
        // to build a real 32k context). Varied tokens exercise real
        // expert routing (vs --fixed's 6 experts), so this reflects the
        // pool-fill / cold-fault / cap-fallback cost a true prompt hits.
        if prefill > 0 {
            let t0 = Date()
            var reportT = Date()
            for i in 0 ..< prefill {
                let tok = (i &* 49_157 &+ 13) % vocab
                _ = try model.forwardAllLayers(inputTokenId: tok, state: state)
                if Date().timeIntervalSince(reportT) > 5 {
                    let done = i + 1
                    let r = Double(done) / Date().timeIntervalSince(t0)
                    dbg(
                        String(
                            format: "prefill %d/%d (%.1f tok/s) allocCount=%d allocGB=%.2f",
                            done, prefill, r, device.bufferAllocCount,
                            Double(device.bufferAllocBytes) / 1e9))
                    reportT = Date()
                }
            }
            let dt = Date().timeIntervalSince(t0)
            print(
                String(
                    format: "[prefill] %d tokens in %.1fs = %.2f tok/s (%.1f ms/tok)",
                    prefill, dt, Double(prefill) / dt, dt / Double(prefill) * 1000))
        }

        if !promptIds.isEmpty {
            let ids = promptIds.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            dbg("prompt-ids: \(ids.count) tokens: \(ids.prefix(20))")
            // Batched prefill next-token logits.
            let pl = try model.forwardPrefillChunk(tokens: ids).toFloatArray()
            // Sequential reference next-token logits (CPU router for determinism).
            let st = model.makeDecodeState()
            if ProcessInfo.processInfo.environment["FFAI_DUMP_ANORM"] == "1" {
                model.dbgAnorm = Tensor.empty(shape: [43 * 4], dtype: .f16, device: Device.shared)
                model.dbgL0 = Tensor.empty(shape: [56], dtype: .f16, device: Device.shared)
            }
            var seq = [Float]()
            for (k, t) in ids.enumerated() {
                st.position = k  // advance RoPE position per token (autoregressive)
                seq = (try model.forwardAllLayers(inputTokenId: t, state: st)).toFloatArray()
            }
            if let da = model.dbgAnorm {
                let a = da.toFloatArray()
                for li in 0 ..< 43 {
                    dbg(
                        String(
                            format: "FFANORM L%d %.5f,%.5f,%.5f,%.5f", li, a[li * 4], a[li * 4 + 1], a[li * 4 + 2],
                            a[li * 4 + 3]))
                }
            }
            if let d0 = model.dbgL0 {
                let v = d0.toFloatArray()
                dbg(
                    String(
                        format: "FFL0 qrnorm=%.5f,%.5f,%.5f,%.5f q=%.5f,%.5f,%.5f,%.5f kv=%.5f,%.5f,%.5f,%.5f", v[12],
                        v[13], v[14], v[15], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]))
                dbg(String(format: "FFL0 heads=%.5f,%.5f,%.5f,%.5f", v[8], v[9], v[10], v[11]))
                if v.count >= 24 {
                    dbg(
                        String(
                            format: "FFL0 blockOut=%.5f,%.5f,%.5f,%.5f newH=%.5f,%.5f,%.5f,%.5f", v[16], v[17], v[18],
                            v[19], v[20], v[21], v[22], v[23]))
                }
                if v.count >= 40 {
                    dbg(
                        String(
                            format:
                                "FFL0FFN moe=%.5f,%.5f,%.5f,%.5f shared=%.5f,%.5f,%.5f,%.5f ffn_out=%.5f,%.5f,%.5f,%.5f after_ffn_hc=%.5f,%.5f,%.5f,%.5f",
                            v[24], v[25], v[26], v[27], v[28], v[29], v[30], v[31], v[32], v[33], v[34], v[35], v[36],
                            v[37], v[38], v[39]))
                }
                if v.count >= 52 {
                    dbg(
                        String(
                            format: "FFL0SHEXP gate=%.5f,%.5f,%.5f,%.5f up=%.5f,%.5f,%.5f,%.5f mid=%.5f,%.5f,%.5f,%.5f",
                            v[40], v[41], v[42], v[43], v[44], v[45], v[46], v[47], v[48], v[49], v[50], v[51]))
                }
                if v.count >= 56 {
                    dbg(
                        String(
                            format: "FFL0EXP0 gate=%.5f,%.5f,%.5f,%.5f up=%.5f,%.5f,%.5f,%.5f", v[48], v[49], v[50],
                            v[51], v[52], v[53], v[54], v[55]))
                }
            }
            func top8(_ a: [Float]) -> [(Int, Float)] {
                return a.enumerated().sorted { $0.element > $1.element }.prefix(8).map { ($0.offset, $0.element) }
            }
            dbg(
                "ratios[0..8]=\(Array(model.layerCompressRatios.prefix(8)))  last=\(model.layerCompressRatios.suffix(2))"
            )
            dbg(
                "compCount L2=\(st.layerStates[2].compCount) L3=\(st.layerStates[3].compCount) L4=\(st.layerStates[4].compCount)"
            )
            let pT = top8(pl); let sT = top8(seq)
            print("[oracle] PREFILL argmax=\(pT[0].0) top8=\(pT.map { "\($0.0):\(String(format: "%.2f", $0.1))" })")
            print("[oracle] SEQDEC  argmax=\(sT[0].0) top8=\(sT.map { "\($0.0):\(String(format: "%.2f", $0.1))" })")
            // ── End-to-end greedy generation: continue from the prompt,
            // feeding back the argmax (sequential decode = the validated
            // correct path). Proves coherent multi-token generation. ──
            var genIDs: [Int] = []
            var lastLogits = seq
            var nNaNGen = 0
            let nGen = max(tokens, 1)
            for step in 0 ..< nGen {
                var mx = 0; var mv = -Float.infinity
                for (j, v) in lastLogits.enumerated() { if v.isNaN { nNaNGen += 1 }; if v > mv { mv = v; mx = j } }
                genIDs.append(mx)
                st.position = ids.count + step
                lastLogits = (try model.forwardAllLayers(inputTokenId: mx, state: st)).toFloatArray()
            }
            print("[oracle] GENERATE (\(nGen) toks, finalPos=\(ids.count + nGen - 1), NaN=\(nNaNGen)) ids=\(genIDs)")
            return
        }

        if validatePrefill > 0 {
            let n = validatePrefill
            let prompt = (0 ..< n).map { ($0 &* 49_157 &+ 13) % vocab }
            // Batched prefill FIRST, on a clean resident pool (running the
            // sequential reference first re-organizes the shared expert pool).
            let t0 = Date()
            let pl = try model.forwardPrefillChunk(tokens: prompt).toFloatArray()
            let dt = Date().timeIntervalSince(t0)
            // WARM second prefill: with FFAI_PREFILL_RESIDENT=1 the expert pool
            // persists, so this run skips the repack/re-read and shows the
            // resident-pool speedup (cold build vs warm reuse).
            let tw = Date()
            _ = try model.forwardPrefillChunk(tokens: prompt).toFloatArray()
            let dtw = Date().timeIntervalSince(tw)
            print(
                String(
                    format: "[prefill-validate] WARM 2nd prefill %d tokens in %.3fs = %.1f tok/s", n, dtw,
                    Double(n) / dtw))
            print(
                String(
                    format: "[prefill-validate] COLD 1st prefill %d tokens in %.3fs = %.1f tok/s", n, dt, Double(n) / dt
                ))
            // FFAI_SKIP_SEQREF=1: skip the O(N) token-by-token reference (which
            // is intolerable at N=8192) — we only want the prefill throughput.
            if ProcessInfo.processInfo.environment["FFAI_SKIP_SEQREF"] == "1" {
                _ = pl
                return
            }
            // Sequential reference: feed the prompt token-by-token; the last
            // call's logits = next-token prediction after the full prompt.
            let seqState = model.makeDecodeState()
            var seqLast = [Float]()
            for (k, t) in prompt.enumerated() {
                dbg("seq ref token \(k)/\(n)")
                seqState.position = k  // per-token RoPE position (batched prefill ropes tokens at 0..N-1)
                seqLast = (try model.forwardAllLayers(inputTokenId: t, state: seqState)).toFloatArray()
            }
            dbg("seq ref done")
            var sMax = 0, sv = -Float.infinity
            for (j, x) in seqLast.enumerated() where x > sv { sv = x; sMax = j }
            var pMax = 0, pv = -Float.infinity
            for (j, x) in pl.enumerated() where x > pv { pv = x; pMax = j }
            // Cosine of the two logit vectors.
            var dot = 0.0, na = 0.0, nb = 0.0
            for i in 0 ..< min(seqLast.count, pl.count) {
                dot += Double(seqLast[i]) * Double(pl[i]); na += Double(seqLast[i] * seqLast[i]);
                nb += Double(pl[i] * pl[i])
            }
            let cos = dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
            print(
                String(
                    format: "[prefill-validate] N=%d  seq_argmax=%d  batched_argmax=%d  cosine=%.5f", n, sMax, pMax, cos
                ))
            print(
                String(
                    format: "[prefill-validate] batched prefill %d tokens in %.3fs = %.1f tok/s", n, dt, Double(n) / dt)
            )
            print(sMax == pMax ? "[prefill-validate] ✅ argmax MATCH" : "[prefill-validate] ❌ argmax mismatch")
            // CAVEAT: the prompt here is SYNTHETIC garbage tokens. On garbage
            // the model is low-confidence (near-tied logits) AND the decode
            // reference is non-deterministic (router-tie bug), so this argmax/
            // cosine comparison is UNRELIABLE — use it for the tok/s number,
            // not correctness. For correctness use `--prompt-ids` with REAL
            // tokenized text (e.g. "The capital of Japan is" → predicts Tokyo).
            print(
                "[prefill-validate] (note: tok/s is meaningful; argmax/cosine on synthetic tokens is NOT — validate correctness via --prompt-ids on real text)"
            )
            return
        }

        var lastTok = bos
        var tpsAll: [Double] = []
        for i in 0 ..< tokens {
            DeepSeekV4Model.resetFfnProf()
            dbg("token \(i) begin (input=\(lastTok))")
            let t0 = Date()
            let logits = try model.forwardAllLayers(inputTokenId: lastTok, state: state)
            let elapsed = Date().timeIntervalSince(t0)
            let host = logits.toFloatArray()
            var maxIdx = 0
            var maxVal: Float = -.infinity
            var nNaN = 0
            for (j, v) in host.enumerated() {
                if v.isNaN { nNaN += 1 }
                if v > maxVal { maxVal = v; maxIdx = j }
            }
            if nNaN > 0 { dbg("token \(i): \(nNaN) NaN logits, maxVal=\(maxVal)") }
            if ProcessInfo.processInfo.environment["FFAI_DBG_LOGITS"] == "1" {
                var sumAbs: Double = 0
                for v in host where v.isFinite { sumAbs += Double(abs(v)) }
                dbg(
                    String(format: "token %d fingerprint: maxVal=%.5f sumAbs=%.3f argmax=%d", i, maxVal, sumAbs, maxIdx)
                )
            }
            let tps = 1.0 / elapsed
            if i > 0 { tpsAll.append(tps) }
            print(String(format: "[bench] token %d took %.3fs (%.2f tps) argmax=%d", i, elapsed, tps, maxIdx))
            if !fixed { lastTok = maxIdx }
        }
        if !tpsAll.isEmpty {
            let mean = tpsAll.reduce(0, +) / Double(tpsAll.count)
            print(String(format: "[bench] sustained mean (tok 1+): %.2f tps", mean))
        }
    }
}

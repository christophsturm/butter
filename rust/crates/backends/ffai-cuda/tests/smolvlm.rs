// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! SmolVLM (Idefics3) vision→text **connector** verified vs HF. This is the one
//! VLM-stitch component not already covered: the SigLIP vision tower and the
//! dense-Llama text model are each verified separately, so the connector —
//! pixel-shuffle (gather a `scale_factor`×`scale_factor` block of patches into
//! one token's channels) + a `modality_projection` linear into the text hidden
//! dim — is the remaining new piece. Pixel-shuffle is exact index math; the
//! projection runs on the verified `matmul`. A full SmolVLM forward then =
//! [verified SigLIP tower] → [this connector] → splice into text → [verified
//! causal Llama], all on the shared op layer.
use ffai_core::{DType, Tensor};
use ffai_cuda::CudaDevice;
use ffai_loader::SafeTensors;
use ffai_ops::matmul;

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }

#[test]
fn smolvlm_connector_vs_hf() {
    let dir = std::env::var("SMOLVLM_DIR").unwrap_or_else(|_| glob_snap().unwrap_or_default());
    let Ok(st) = SafeTensors::open_dir(&dir) else { eprintln!("no model at {dir} — skipping"); return; };
    let Some(dev) = CudaDevice::create().expect("cuda") else { eprintln!("no CUDA — skip"); return; };
    let d = dev.as_ref();

    let (vdim, sf, txt) = (768usize, 4usize, 576usize); // vision hid, scale_factor, text hid
    let np = 1024usize; let grid = 32usize; // 32×32 patches
    let og = grid / sf; // 8 → 64 output tokens
    let n_tok = og * og; // 64
    let cin = vdim * sf * sf; // 12288

    let up = |v: &[f32], sh: Vec<usize>| -> Tensor { Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32) };
    let dl = |t: &Tensor, n: usize| -> Vec<f32> { let mut b = vec![0u8; n * 4]; d.download(t.buffer.as_ref(), &mut b).unwrap(); fb(&b) };

    // deterministic vision-tower output [1024, 768]
    let x: Vec<f32> = (0..np * vdim).map(|i| (0.01 * i as f32).sin()).collect();

    // pixel-shuffle: out[tok=h4*og+w2][e4], e4 → h_sub=e4/(vdim*sf), w_sub=(e4 % (vdim*sf))/vdim, e=e4%vdim
    // source patch (h=h4*sf+h_sub, w=w2*sf+w_sub): x[(h*grid + w)*vdim + e]
    let mut shuf = vec![0.0f32; n_tok * cin];
    for h4 in 0..og { for w2 in 0..og {
        let tok = h4 * og + w2;
        for e4 in 0..cin {
            let h_sub = e4 / (vdim * sf);
            let w_sub = (e4 % (vdim * sf)) / vdim;
            let e = e4 % vdim;
            let h = h4 * sf + h_sub; let w = w2 * sf + w_sub;
            shuf[tok * cin + e4] = x[(h * grid + w) * vdim + e];
        }
    }}

    // modality_projection (Linear, no bias): [txt, cin] @ shuf[n_tok, cin] → [n_tok, txt]
    let proj = st.tensor_f32("model.connector.modality_projection.proj.weight").unwrap().0;
    let out = dl(&matmul(d, &up(&proj, vec![txt, cin]), &up(&shuf, vec![n_tok, cin])).unwrap(), n_tok * txt);

    let want0 = [1.56379f32, 0.48131, -0.04509, -1.15257, 1.72148];
    let want63 = [-0.73231f32, -0.52461, 1.12685, 1.35081, -2.54256];
    let mut e = 0.0f32;
    for i in 0..5 { e = e.max((out[i] - want0[i]).abs()); }
    for i in 0..5 { e = e.max((out[63 * txt + i] - want63[i]).abs()); }
    let sum: f32 = out.iter().sum();
    eprintln!("SmolVLM connector on CUDA: out[0,:5]={:?}", &out[..5]);
    eprintln!("  out[63,:5]={:?}  sum={sum:.3} (HF=79.852)  max|Δ|={e:.3e}", &out[63 * txt..63 * txt + 5]);
    assert!(e < 2e-3, "SmolVLM connector mismatch vs HF: {e:.3e}");
    assert!((sum - 79.852).abs() < 0.5, "connector sum off: {sum}");
    eprintln!("✅ SmolVLM connector (pixel-shuffle + modality projection) matches HF on the shared engine (GB10 sm_121) — VLM stitch component verified.");
}

fn glob_snap() -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}

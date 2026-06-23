// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! GGUF regression GATE — the single runnable check that locks in the M5 GGUF
//! lane. For every supported model it loads the GGUF end-to-end and generates a
//! completion of "The capital of France is" through BOTH inference paths:
//!
//!   * f32          — `GgufModel::open`    (dequant-to-upload, the reference)
//!   * resident-Q8  — `GgufModel::open_q8` (weights kept Q8, `gemv_q8` decode —
//!                                          20-57× faster, bit-identical)
//!
//! and asserts, per model:
//!   1. COHERENCE  — both continuations contain "Paris".
//!   2. AGREEMENT  — the f32 and Q8 greedy token streams are IDENTICAL over the
//!                   shared decode budget (resident-Q8 is lossless here).
//!
//! Models covered (the production GGUF matrix):
//!   * Qwen2.5-1.5B-Instruct  Q8_0   (single file, BPE tokenizer)
//!   * Qwen2.5-7B-Instruct    Q4_K_M (2-part split, k-quant super-blocks)
//!   * Phi-3.1-mini-128k      Q4_K_M (FUSED qkv/ffn tensors, SPM tokenizer)
//!
//! This is the regression gate: future changes that silently break GGUF load,
//! coherence, or f32↔Q8 agreement fail HERE. It is backend-agnostic by
//! construction (host loader + registry ops), so the same path runs on
//! CUDA/Vulkan; the gate runs on macOS + Metal and SKIPS (passes) cleanly if no
//! Metal device or the model file is absent.
//!
//! Runtime note: the f32 path on the 7B Q4_K_M is the slow dequant-to-upload
//! path (~13 s/token), so the f32 baseline uses a SHORT shared budget while the
//! Q8 path — the recommended decode path — runs the full continuation. The
//! f32↔Q8 agreement assert compares only the shared (short) prefix.

use ffai_loader::gguf::Gguf;
use ffai_metal::MetalDevice;
use ffai_models::gguf_tokenizer::GgufTokenizer;
use ffai_models::llama::GgufModel;
use ffai_runtime::{generate, Sampling, StopOn};
use std::time::Instant;

/// One row of the GGUF regression matrix.
struct ModelCase {
    /// Human label for log lines / assert messages.
    label: &'static str,
    /// Env var that overrides the model path (CI / alternate locations).
    env: &'static str,
    /// Default local path (part-1 of a split is fine — the loader opens the rest).
    default_path: &'static str,
    /// Shared f32↔Q8 agreement budget. The f32 path runs this many decode
    /// steps; Q8 runs `full_budget` but agreement compares only this prefix.
    /// Keep small for models whose f32 path is the slow dequant-to-upload path.
    agree_budget: usize,
    /// Q8 decode budget (the recommended path; cheap, so run a real continuation).
    full_budget: usize,
}

const CASES: &[ModelCase] = &[
    ModelCase {
        label: "Qwen2.5-1.5B-Q8_0",
        env: "QWEN25_GGUF",
        default_path: "/Users/tom/models/qwen2.5-1.5b-instruct-q8_0.gguf",
        agree_budget: 12,
        full_budget: 12,
    },
    ModelCase {
        label: "Qwen2.5-7B-Q4_K_M",
        env: "QWEN25_7B_GGUF",
        default_path: "/Users/tom/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        // f32 on the 7B is ~13 s/token (dequant-to-upload); keep the shared
        // agreement budget tiny so the gate stays fast.
        agree_budget: 4,
        full_budget: 16,
    },
    ModelCase {
        label: "Phi-3.1-mini-Q4_K_M",
        env: "PHI3_GGUF",
        default_path: "/Users/tom/models/Phi-3.1-mini-128k-instruct-Q4_K_M_2.gguf",
        agree_budget: 12,
        full_budget: 16,
    },
];

const PROMPT: &str = "The capital of France is";
const CAP: usize = 256;

/// Pick the EOS id robustly across tokenizer flavors (Qwen BPE vs Phi-3 SPM).
fn eos_id(tok: &GgufTokenizer) -> Option<u32> {
    tok.token_id("<|endoftext|>")
        .or_else(|| tok.token_id("<|end|>"))
        .or_else(|| tok.token_id("</s>"))
}

/// Run `model` greedily for `budget` decode steps and return (tokens, text, decode_tok/s).
fn run(
    dev: &dyn ffai_core::Device,
    model: &GgufModel,
    tok: &GgufTokenizer,
    prompt: &[u32],
    eos: Option<u32>,
    budget: usize,
) -> (Vec<u32>, String, f32) {
    let stop = StopOn { max_new: budget, eos };
    // Warm the JIT so the first kernel compile doesn't pollute tok/s.
    let _ = model.step(dev, prompt[0], 0).expect("warmup step");
    let t = Instant::now();
    let mut decode_steps = 0usize;
    let out = generate(prompt, &stop, &Sampling::Greedy, 0, |token, pos| {
        if pos + 1 > prompt.len() {
            decode_steps += 1;
        }
        model.step(dev, token, pos).expect("decode step")
    });
    let tps = decode_steps as f32 / t.elapsed().as_secs_f32().max(1e-6);
    let text = tok.decode(&out);
    (out, text, tps)
}

/// THE GATE. One test, the whole GGUF matrix × {f32, resident-Q8}.
#[test]
fn gguf_regression_gate_all_models_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping GGUF regression gate");
        return;
    };
    let d = dev.as_ref();
    eprintln!("═══ GGUF REGRESSION GATE — device: {} ═══", dev.name());

    let mut ran = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for case in CASES {
        let path = std::env::var(case.env).unwrap_or_else(|_| case.default_path.to_string());
        if !std::path::Path::new(&path).exists() {
            eprintln!("── {}: model not found at {path} — SKIP", case.label);
            continue;
        }
        ran += 1;
        eprintln!("\n── {} ──────────────────────────────", case.label);

        let g = Gguf::open(&path).expect("open gguf");
        let tok = GgufTokenizer::from_gguf(&g).expect("tokenizer");
        let prompt = tok.encode(PROMPT);
        assert!(!prompt.is_empty(), "{}: tokenizer produced no ids", case.label);
        let eos = eos_id(&tok);
        eprintln!(
            "   arch={:?} vocab={} prompt_ids={:?}",
            g.meta_str("general.architecture"),
            tok.vocab_size(),
            prompt
        );

        // ── f32 reference path (shared agreement budget) ──────────────────
        let m_f32 = GgufModel::open(d, &path, CAP).expect("open f32");
        let (out_f32, text_f32, tps_f32) =
            run(d, &m_f32, &tok, &prompt, eos, case.agree_budget);
        drop(m_f32); // free f32 weights before loading Q8 (peak-mem hygiene)
        eprintln!("   f32  : {tps_f32:6.2} tok/s  {text_f32:?}");

        // ── resident-Q8 path (recommended; full continuation) ─────────────
        let m_q8 = GgufModel::open_q8(d, &path, CAP).expect("open_q8");
        let (out_q8, text_q8, tps_q8) = run(d, &m_q8, &tok, &prompt, eos, case.full_budget);
        eprintln!("   Q8   : {tps_q8:6.2} tok/s  {text_q8:?}");
        eprintln!("   Q8 speedup vs f32: {:.1}×", tps_q8 / tps_f32.max(1e-6));

        // ── ASSERT 1: coherence on BOTH paths ─────────────────────────────
        if !text_f32.contains("Paris") {
            failures.push(format!("{}: f32 lost coherence — {text_f32:?}", case.label));
        }
        if !text_q8.contains("Paris") {
            failures.push(format!("{}: Q8 lost coherence — {text_q8:?}", case.label));
        }

        // ── ASSERT 2: f32↔Q8 token agreement over the shared prefix ───────
        let n = case.agree_budget.min(out_f32.len()).min(out_q8.len());
        if out_f32[..n] != out_q8[..n] {
            failures.push(format!(
                "{}: f32↔Q8 token disagreement over {n} steps\n      f32={:?}\n      q8 ={:?}",
                case.label,
                &out_f32[..n],
                &out_q8[..n],
            ));
        } else {
            eprintln!("   ✅ coherent (f32 & Q8) + f32↔Q8 token streams identical ({n} steps)");
        }
    }

    if ran == 0 {
        eprintln!("\n⚠️  no GGUF models present — gate vacuously passed (set *_GGUF env vars)");
        return;
    }
    assert!(
        failures.is_empty(),
        "GGUF regression gate FAILED ({} model(s) checked):\n  - {}",
        ran,
        failures.join("\n  - "),
    );
    eprintln!("\n═══ ✅ GGUF REGRESSION GATE PASSED — {ran} model(s) × {{f32, resident-Q8}} ═══");
}

/// The recommended entrypoint [`GgufModel::load`] (prefers resident-Q8, falls
/// back to f32) must load + generate coherently. Quick smoke on the smallest
/// model so the default GGUF inference path is itself regression-covered.
#[test]
fn gguf_recommended_load_entrypoint_metal() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    let d = dev.as_ref();
    let path = std::env::var("QWEN25_GGUF")
        .unwrap_or_else(|_| "/Users/tom/models/qwen2.5-1.5b-instruct-q8_0.gguf".to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("model not found at {path} — skipping");
        return;
    }
    let g = Gguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(&g).expect("tokenizer");
    let prompt = tok.encode(PROMPT);
    let eos = eos_id(&tok);

    // `load` should pick the resident-Q8 path here (Q8_0 repacks cleanly).
    let model = GgufModel::load(d, &path, CAP).expect("recommended load");
    let (_, text, tps) = run(d, &model, &tok, &prompt, eos, 12);
    eprintln!("GgufModel::load → {tps:.2} tok/s  {text:?}");
    assert!(text.contains("Paris"), "recommended load lost coherence: {text:?}");
    eprintln!("✅ GgufModel::load (recommended resident-Q8 entrypoint) coherent.");
}

/// QUANT-COVERAGE AUDIT. For every supported model, enumerate the distinct ggml
/// quant types present in its 2-D matmul tensors and spot-check that
/// `Gguf::q8_repack` (the resident-Q8 repack used by `open_q8`) succeeds on one
/// tensor of EACH type. This proves the resident-Q8 path covers every
/// quant×model combo we ship (Q8_0, Q4_K, Q5_K, Q6_K) — no Metal device needed.
#[test]
fn gguf_resident_q8_quant_coverage_audit() {
    use std::collections::BTreeMap;
    let mut any = false;
    for case in CASES {
        let path = std::env::var(case.env).unwrap_or_else(|_| case.default_path.to_string());
        if !std::path::Path::new(&path).exists() {
            eprintln!("── {}: not found — SKIP", case.label);
            continue;
        }
        any = true;
        let g = Gguf::open(&path).expect("open gguf");
        // One representative 2-D tensor name per distinct quant type.
        let mut rep: BTreeMap<String, String> = BTreeMap::new();
        for name in g.tensor_names() {
            let t = g.tensor(name).unwrap();
            if t.dims.len() == 2 {
                rep.entry(format!("{:?}", t.ggml_type)).or_insert_with(|| name.clone());
            }
        }
        eprintln!("── {} — 2-D quant types: {:?}", case.label, rep.keys().collect::<Vec<_>>());
        for (ty, name) in &rep {
            match g.q8_repack(name) {
                Ok((_, _, m, k)) => eprintln!("   ✅ {ty:<10} q8_repack ok ({name}, [{m}×{k}])"),
                Err(e) => panic!("   ❌ {ty} q8_repack FAILED on {name}: {e}"),
            }
        }
    }
    if !any {
        eprintln!("⚠️ no models present — audit vacuously passed");
    }
    eprintln!("✅ resident-Q8 quant-coverage audit passed (all present quant types repack).");
}

// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Capability probe demo — what a checkpoint can ACTUALLY do, from its tensors.
use ffai_loader::probe;

fn snap(repo: &str) -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--{repo}/snapshots", std::env::var("HOME").ok()?);
    std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok()).next().map(|e| e.path().to_string_lossy().into_owned())
}

#[test]
fn probe_real_models() {
    let models = [
        ("gpt2", "gpt2"),
        ("HuggingFaceTB--SmolVLM-256M-Instruct", "SmolVLM"),
        ("openai--whisper-base", "Whisper"),
        ("unsloth--gemma-2-2b-it", "Gemma-2"),
        ("microsoft--phi-1_5", "Phi-1.5"),
    ];
    println!("\n{:<12} {:>6} {:>10}   detected(text/vis/aud)  declared          notes", "model", "tens", "params");
    println!("{}", "-".repeat(96));
    for (repo, label) in models {
        let Some(dir) = snap(repo) else { println!("{label:<12} (not cached)"); continue };
        match probe(&dir) {
            Ok(p) => {
                let det = format!("{}/{}/{}", b(p.detected.text), b(p.detected.vision), b(p.detected.audio));
                let dec = p.config_model_type.clone().unwrap_or_else(|| "?".into());
                println!("{label:<12} {:>6} {:>9.0}M   {det:<22}  {dec:<16}  {}",
                    p.n_tensors, p.n_params as f64 / 1e6, p.warnings.first().cloned().unwrap_or_default());
            }
            Err(e) => println!("{label:<12} probe error: {e}"),
        }
    }
    println!();
}

fn b(v: bool) -> &'static str { if v { "✓" } else { "·" } }

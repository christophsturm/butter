// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Butter CLI. With no subcommand, reports the backends compiled into this
//! build and any live devices (the original skeleton behavior, kept as a
//! quick "is this build wired right" check). `generate` is the real
//! generation entrypoint: load a GGUF, batched chunked prefill the prompt
//! (`wh_butter::generate::run` -> `<arch>::prefill`), then decode
//! token-by-token, streaming output and reporting prefill/decode tok/s
//! separately.

use wh_butter::generate::GenerateOpts;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("generate") => generate_cmd(&args[1..]),
        Some("help" | "--help" | "-h") => {
            print_usage();
        }
        Some(other) => {
            eprintln!("butter: unrecognized subcommand {other:?}\n");
            print_usage();
            std::process::exit(2);
        }
        None => report_backends(),
    }
}

fn print_usage() {
    println!("Butter — Melt away latency with ultra-smooth local inference (Rust engine)");
    println!();
    println!("USAGE:");
    println!("    butter                       report compiled backends + live devices");
    println!("    butter generate [OPTIONS]    load a GGUF and generate text");
    println!();
    println!("generate OPTIONS:");
    println!("    --model <path>       GGUF path (required)");
    println!("    --prompt <text>      prompt text (default: a short built-in prompt)");
    println!("    --prompt-file <path> read the prompt from a file instead of --prompt");
    println!("    --max-new <n>        max tokens to generate (default: 128)");
    println!("    --chunk <n>          prefill chunk size, 0 = model's bench-validated default (default: 0)");
    println!("    --seed <n>           sampling seed (default: 0)");
    println!("    --greedy             greedy decoding (default: on — this port is greedy-only for now)");
    println!("    --no-stream          don't print tokens as they're generated");
}

fn report_backends() {
    println!("Butter — Melt away latency with ultra-smooth local inference (Rust engine)");
    println!("compiled backends: {:?}", wh_butter::compiled_backends());

    let devices = wh_butter::devices();
    if devices.is_empty() {
        println!("live devices: none (backends are stubs — Device impls pending)");
    } else {
        println!("live devices:");
        for d in &devices {
            println!("  [{}] {}", d.backend().as_str(), d.name());
        }
    }
}

const DEFAULT_PROMPT: &str = "The capital of France is";

fn generate_cmd(args: &[String]) {
    let mut model: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut prompt_file: Option<String> = None;
    let mut max_new = 128usize;
    let mut chunk = 0usize;
    let mut seed = 0u64;
    let mut stream = true;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model = args.get(i).cloned(); }
            "--prompt" => { i += 1; prompt = args.get(i).cloned(); }
            "--prompt-file" => { i += 1; prompt_file = args.get(i).cloned(); }
            "--max-new" => { i += 1; max_new = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(max_new); }
            "--chunk" => { i += 1; chunk = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(chunk); }
            "--seed" => { i += 1; seed = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(seed); }
            "--greedy" => {}
            "--no-stream" => { stream = false; }
            other => {
                eprintln!("butter generate: unrecognized flag {other:?}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let Some(model) = model else {
        eprintln!("butter generate: --model <path> is required");
        std::process::exit(2);
    };

    let prompt = match (prompt, prompt_file) {
        (Some(_), Some(_)) => {
            eprintln!("butter generate: pass only one of --prompt / --prompt-file");
            std::process::exit(2);
        }
        (Some(p), None) => p,
        (None, Some(path)) => match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("butter generate: reading --prompt-file {path:?}: {e}");
                std::process::exit(1);
            }
        },
        (None, None) => DEFAULT_PROMPT.to_string(),
    };

    let devices = wh_butter::devices();
    let Some(dev) = devices.first() else {
        eprintln!(
            "butter generate: no live device (compiled backends: {:?}) — build with \
             --features cuda/metal/vulkan matching your hardware",
            wh_butter::compiled_backends()
        );
        std::process::exit(1);
    };
    eprintln!("butter generate: device [{}] {}", dev.backend().as_str(), dev.name());

    let opts = GenerateOpts {
        max_new,
        chunk,
        sampling: wh_butter::generate::Sampling::Greedy,
        seed,
        stream,
    };

    let t0 = std::time::Instant::now();
    let report = match wh_butter::generate::run(dev.as_ref(), &model, &prompt, &opts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("butter generate: {e}");
            std::process::exit(1);
        }
    };
    let wall = t0.elapsed().as_secs_f64();

    if !opts.stream {
        println!("{}", report.text);
    }
    eprintln!();
    eprintln!("architecture:     {}", report.architecture);
    eprintln!("prompt tokens:    {}", report.prompt_tokens);
    eprintln!("generated tokens: {}", report.generated_tokens);
    eprintln!(
        "prefill: {:.3}s => {:.2} tok/s",
        report.prefill_secs, report.prefill_tok_s
    );
    eprintln!(
        "decode:  {:.3}s => {:.2} tok/s",
        report.decode_secs, report.decode_tok_s
    );
    eprintln!("end-to-end wall:  {:.3}s", wall);
}

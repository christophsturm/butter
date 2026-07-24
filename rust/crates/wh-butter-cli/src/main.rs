// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Butter CLI — skeleton. For now it reports the backends compiled into this
//! build and any live devices, so the modular feature wiring is visible.

fn main() {
    println!("Butter — Melt away latency with ultra-smooth local inference (Rust engine, skeleton)");
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

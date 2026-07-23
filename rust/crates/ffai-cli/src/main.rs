// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! FFAI CLI — skeleton. For now it reports the backends compiled into this
//! build and any live devices, so the modular feature wiring is visible.

fn main() {
    println!("FFAI — F*cking Fast AI (Rust engine, skeleton)");
    println!("compiled backends: {:?}", ffai::compiled_backends());

    let devices = ffai::devices();
    if devices.is_empty() {
        println!("live devices: none (backends are stubs — Device impls pending)");
    } else {
        println!("live devices:");
        for d in &devices {
            println!("  [{}] {}", d.backend().as_str(), d.name());
        }
    }
}

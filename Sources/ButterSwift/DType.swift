// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Numeric types Butter tensors can hold. Matches iron's DType subset
// we actually use for inference.

import Foundation

public enum DType: String, Sendable, Hashable, Codable {
    // ── GPU-dispatchable types ────────────────────────────────────────
    case f32, f16, bf16, i32, u32, i8, u8
    // ── Metadata / non-dispatch types ────────────────────────────────
    // 64-bit ints appear in upstream HF checkpoints as BatchNorm
    // running counters (`num_batches_tracked` is int64 by convention)
    // and as positional-index lookup tables. No Butter inference path
    // dispatches a kernel against them — they're carried through the
    // loader as opaque buffers so model code can read them as bytes
    // (or, more commonly, ignore them entirely).
    case i64, u64

    public var byteSize: Int {
        switch self {
        case .i64, .u64: return 8
        case .f32, .i32, .u32: return 4
        case .f16, .bf16: return 2
        case .i8, .u8: return 1
        }
    }

    /// Parse a SafeTensors dtype string (e.g. "F32", "F16", "BF16").
    public static func fromSafeTensors(_ s: String) -> DType? {
        switch s.uppercased() {
        case "F32": return .f32
        case "F16": return .f16
        case "BF16": return .bf16
        case "I32": return .i32
        case "U32": return .u32
        case "I64": return .i64
        case "U64": return .u64
        case "I8": return .i8
        case "U8": return .u8
        // F8_E4M3 (float8_e4m3fn) has no dedicated Butter dtype — read
        // its raw bytes as opaque `.u8`. Two call sites decode them
        // deliberately: NVFP4 per-block micro-scales are ALREADY u8
        // E4M3-encoded by convention (matches `Nvfp4Linear.scales`),
        // so those pass straight through; NemotronH-Lightning's static
        // FP8 mamba mixer projections are genuine E4M3-encoded weight
        // values and get host-side dequantized to bf16 at load time
        // (see `NemotronHNvfp4.swift`). Added for the Nemotron-3.5-
        // Lightning-30B-A3B-NVFP4 port — see `NemotronHNvfp4.swift`.
        case "F8_E4M3": return .u8
        default: return nil
        }
    }

    /// Suffix used in iron-emit kernel names ("rms_norm_f16", etc.)
    public var kernelSuffix: String { rawValue }
}

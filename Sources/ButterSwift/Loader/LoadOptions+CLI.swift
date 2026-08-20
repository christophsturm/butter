// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed as an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Shared CLI mapping for generate + serve load knobs.

import Foundation

extension LoadOptions {
    public enum CLIError: Error, CustomStringConvertible, Equatable {
        case unknownKV(String)
        case unknownAURA(String)
        case unknownAuraPath(String)
        case windowKeepNeedsSize
        case maxContextMustBePositive

        public var description: String {
            switch self {
            case .unknownKV(let s):
                return "Unknown --kv-cache \"\(s)\". Use \"raw\", \"affine8\", \"affine4\", or any \"auraNvM\" recipe."
            case .unknownAURA(let s):
                return "Unknown AURA recipe \"\(s)\". Try \"aura\", \"aura4\", \"aura4v2\", \"aura3\", \"aura8\"."
            case .unknownAuraPath(let s):
                return "Unknown --aura-decode-path \"\(s)\". Use \"compressed\" or \"dequant-mirror\"."
            case .windowKeepNeedsSize:
                return "--kv-window-keep requires --kv-window-size to be set."
            case .maxContextMustBePositive:
                return "--max-context must be > 0."
            }
        }
    }

    /// Build load options from serve/generate CLI flags.
    public static func fromCLI(
        kvCache: String? = nil,
        kvWindowSize: Int? = nil,
        kvWindowKeep: Int? = nil,
        auraDecodePath: String? = nil,
        maxContextLength: Int? = nil
    ) throws -> LoadOptions {
        var loadOpts = LoadOptions()
        let rawKVKind = (kvCache ?? "raw").lowercased()
        switch rawKVKind {
        case "raw":
            loadOpts.kvCache = .raw
        case "affine8":
            loadOpts.kvCache = .affineQuantized(bits: 8, groupSize: 64)
        case "affine4":
            loadOpts.kvCache = .affineQuantized(bits: 4, groupSize: 16)
        case _ where rawKVKind.hasPrefix("aura"):
            guard let scheme = AURAScheme.parse(rawKVKind) else {
                throw CLIError.unknownAURA(rawKVKind)
            }
            loadOpts.kvCache = .auraQuantized(scheme: scheme)
        default:
            throw CLIError.unknownKV(kvCache ?? "")
        }

        if let size = kvWindowSize, size > 0 {
            loadOpts.kvEviction = .window(maxSize: size, keep: kvWindowKeep ?? 0)
        } else if (kvWindowKeep ?? 0) != 0 {
            throw CLIError.windowKeepNeedsSize
        }

        if let raw = auraDecodePath?.lowercased() {
            switch raw {
            case "compressed":
                loadOpts.auraDecodePath = .compressed
            case "dequant-mirror", "dequant_mirror", "mirror":
                loadOpts.auraDecodePath = .dequantMirror
            default:
                throw CLIError.unknownAuraPath(auraDecodePath ?? "")
            }
        }

        if let n = maxContextLength {
            guard n > 0 else { throw CLIError.maxContextMustBePositive }
            loadOpts.maxContextLength = n
        }
        return loadOpts
    }
}

// swift-tools-version: 6.1
//
// Butter — Melt away latency with ultra-smooth local inference.
//
// Apple Silicon LLM inference library built on pre-compiled Metal kernels
// generated from the Iron Rust DSL.
// See planning/plan.md and planning/architecture.md.

import PackageDescription

let package = Package(
    name: "Butter",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "ButterSwift",
            targets: ["ButterSwift"]
        ),
        .library(
            name: "IronSwift",
            targets: ["IronSwift"]
        ),
        .executable(
            name: "butter",
            targets: ["ButterCLI"]
        ),
    ],
    dependencies: [
        // Hugging Face Hub client for model snapshot download/cache.
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        // Tokenizer loading (AutoTokenizer.from(modelFolder:)).
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // CLI argument parsing for butter.
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        // Pre-compiled Metal kernels + Swift dispatch wrappers.
        // Resources are produced at build time by `iron build --emit all`
        // (the iron-cli binary in the sibling Iron workspace)
        // and live under Resources/. Generated/ contains the typed Swift
        // wrappers, also produced by `iron build`.
        .target(
            name: "IronSwift",
            path: "Sources/IronSwift",
            resources: [
                .copy("Resources")
            ]
        ),

        // Main inference library: Tensor, Module system, model families,
        // KV cache, sampling, generate loop, HF download, tokenizer.
        .target(
            name: "ButterSwift",
            dependencies: [
                "IronSwift",
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/ButterSwift",
            swiftSettings: [
                // Opt into Accelerate's ILP64 CBLAS interface
                // (`cblas_sgemm` / `cblas_sgemv` were deprecated in
                // macOS 13.3 in favour of this set of headers). Without
                // it every audio/vision DSP file that wraps a BLAS call
                // gets a deprecation warning per build pass. `_ILP64`
                // is required by the new header to enable the 64-bit
                // size_t-shaped BLAS_INDEX type.
                //
                // These must reach clang (which compiles the imported
                // Accelerate headers), so `.unsafeFlags(["-Xcc", ...])`
                // is required — `.define(...)` only forwards to swiftc
                // for `#if SWIFT_…` and does nothing for the C side.
                .unsafeFlags([
                    "-Xcc",
                    "-DACCELERATE_NEW_LAPACK",
                    "-Xcc",
                    "-DACCELERATE_LAPACK_ILP64",
                ])
            ]
        ),

        // CLI: butter --model <id-or-path> --prompt "..."
        .executableTarget(
            name: "ButterCLI",
            dependencies: [
                "ButterSwift",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/ButterCLI"
        ),

        // Shared test helpers (model-load lock, image / audio /
        // checkpoint resolution, coherent-output + content-recognition
        // assertions). Lives in its own target so both ButterTests and
        // ModelIntegrationTests can depend on it without source duplication.
        // Uses only ButterSwift's public API — no `@testable` import.
        // Fixtures live at `Tests/Resources/` and are resolved by the
        // helpers via `#filePath`-relative paths (not SwiftPM bundles).
        .target(
            name: "TestHelpers",
            dependencies: ["ButterSwift"],
            path: "Tests/Helpers"
        ),

        // Tests
        .testTarget(
            name: "IronSwiftTests",
            dependencies: ["IronSwift"],
            path: "Tests/IronSwiftTests"
        ),
        .testTarget(
            name: "ButterTests",
            dependencies: ["ButterSwift", "TestHelpers"],
            path: "Tests/ButterTests",
            // Laguna config fixtures are resolved via #filePath-relative
            // paths (repo convention), not SwiftPM resource bundles.
            exclude: ["Fixtures"]
        ),
        .testTarget(
            name: "ModelIntegrationTests",
            dependencies: ["ButterSwift", "TestHelpers"],
            path: "Tests/ModelIntegrationTests"
        ),
    ]
)

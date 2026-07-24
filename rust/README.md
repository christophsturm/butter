# Butter — Rust engine

The cross-platform half of Butter. One Rust core behind a single `Device`
trait; backends (CUDA, Vulkan, ROCm, Metal) are independent crates picked at
build time by cargo feature.

> **The Swift engine (repo root) is the primary Apple path** — native Metal,
> ships to iPhone/iPad/Mac, fast. It is *maintained as a first-class engine*,
> not replaced. This Rust engine is the path to everything Swift can't reach:
> NVIDIA (CUDA), AMD (ROCm), and portable GPUs (Vulkan). Both share the same
> kernels via the **iron** IR, so a kernel is written once and lowered to
> MSL *and* CUDA/SPIR-V.

## Layout

```
rust/
  crates/
    wh-butter-core/        Device trait, Tensor, DType, Binding — the one seam
    wh-butter-ops/         semantic ops (rms_norm, matmul, …) → Kernel → dispatch
    wh-butter-models/      model forward passes (ported from swift Models/, validated against it)
    wh-butter-loader/      GGUF / SafeTensors / HF
    wh-butter-runtime/     KV cache, sampler, decode loop
    wh-butter/          umbrella: backend selection + device enumeration
    wh-butter-cli/         `butter` binary
    backends/
      wh-butter-cuda/      wraps wh-iron-runtime CudaDevice (NVRTC → PTX → driver)
      wh-butter-metal/     metal-rs (cross-validation on Mac; Swift is primary on Apple)
      wh-butter-vulkan/    SPIR-V (portable; pending iron SPIR-V target)
```

## Build

```sh
# Mac dev box (default = metal stub)
cargo run -p wh-butter-cli

# CUDA host (e.g. the GB10 / DGX Spark)
cargo run -p wh-butter-cli --no-default-features --features cuda
```

## How sharing works

The boundary is exactly two things: the **`Device` trait** (`wh-butter-core`) and
the **op→kernel dispatch** (`wh-butter-ops`). Everything above them — models,
loaders, KV cache, sampler — is plain Rust that never names a GPU API, so it
runs on every backend unchanged. Everything below is a thin `Device` impl.
Kernels are already shared by iron.

Adding a backend (e.g. Vulkan) is **one crate**, not a fork — that's the
whole point of the seam.

## iron dependency

Canonical pointer is the `feature/cuda-backend` branch of the iron repo
(the cuda+metal toolchain), so a fresh clone is self-contained. Local
co-development overrides it to a sibling `../../iron-cuda` checkout via
the `[patch]` in `Cargo.toml` — instant, no fetch.

## Status

Skeleton: the `Device` trait + Tensor + modular backend crates compile and the
CLI enumerates compiled backends. Backend `Device` impls and the model port
land next. The CUDA kernel corpus already passes 100% (4164/4164) bit-accurate
on GB10 in the iron repo — this engine wires those kernels into real
inference.

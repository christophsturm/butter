# FFAI — Rust engine

The cross-platform half of FFAI. One Rust core behind a single `Device`
trait; backends (CUDA, Vulkan, ROCm, Metal) are independent crates picked at
build time by cargo feature.

> **The Swift engine (repo root) is the primary Apple path** — native Metal,
> ships to iPhone/iPad/Mac, fast. It is *maintained as a first-class engine*,
> not replaced. This Rust engine is the path to everything Swift can't reach:
> NVIDIA (CUDA), AMD (ROCm), and portable GPUs (Vulkan). Both share the same
> kernels via the **metaltile** IR, so a kernel is written once and lowered to
> MSL *and* CUDA/SPIR-V.

## Layout

```
rust/
  crates/
    ffai-core/        Device trait, Tensor, DType, Binding — the one seam
    ffai-ops/         semantic ops (rms_norm, matmul, …) → Kernel → dispatch
    ffai-models/      model forward passes (ported from swift Models/, validated against it)
    ffai-loader/      GGUF / SafeTensors / HF
    ffai-runtime/     KV cache, sampler, decode loop
    ffai/             umbrella: backend selection + device enumeration
    ffai-cli/         `ffai` binary
    backends/
      ffai-cuda/      wraps metaltile-runtime CudaDevice (NVRTC → PTX → driver)
      ffai-metal/     metal-rs (cross-validation on Mac; Swift is primary on Apple)
      ffai-vulkan/    SPIR-V (portable; pending metaltile SPIR-V target)
```

## Build

```sh
# Mac dev box (default = metal stub)
cargo run -p ffai-cli

# CUDA host (e.g. the GB10 / DGX Spark)
cargo run -p ffai-cli --no-default-features --features cuda
```

## How sharing works

The boundary is exactly two things: the **`Device` trait** (`ffai-core`) and
the **op→kernel dispatch** (`ffai-ops`). Everything above them — models,
loaders, KV cache, sampler — is plain Rust that never names a GPU API, so it
runs on every backend unchanged. Everything below is a thin `Device` impl.
Kernels are already shared by metaltile.

Adding a backend (e.g. Vulkan) is **one crate**, not a fork — that's the
whole point of the seam.

## metaltile dependency

Canonical pointer is the `feature/cuda-backend` branch of the metaltile repo
(the cuda+metal toolchain), so a fresh clone is self-contained. Local
co-development overrides it to a sibling `../../metaltile-cuda` checkout via
the `[patch]` in `Cargo.toml` — instant, no fetch.

## Status

Skeleton: the `Device` trait + Tensor + modular backend crates compile and the
CLI enumerates compiled backends. Backend `Device` impls and the model port
land next. The CUDA kernel corpus already passes 100% (4164/4164) bit-accurate
on GB10 in the metaltile repo — this engine wires those kernels into real
inference.

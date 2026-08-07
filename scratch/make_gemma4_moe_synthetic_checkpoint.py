#!/usr/bin/env python3
# Synthetic Gemma 4 MoE (26B-A4B-shaped) checkpoint generator for #105
# (per-token transient-allocation investigation in Gemma4MoEFFN.forward /
# the gemma4 MoE decode path).
#
# The real ~/models/gemma-4-26b-a4b-4bit checkpoint is 15GB+ and is
# EXPLICITLY OFF LIMITS for this investigation (hard safety rule — loading
# it crashed the machine twice already). This script writes a tiny,
# random-weighted checkpoint with the SAME tensor names / shapes / config
# keys the real checkpoint has, at knob-controlled (tiny by default) dims,
# so Butter's real Gemma4MoE loader + forward path runs unmodified and
# produces real (if numerically meaningless) allocation behavior we can
# measure with BUTTER_ALLOC_TRACE=1. Output is gibberish; only the
# allocation *volume* is under test here, not correctness.
#
# Tensor names/shapes were read directly off Sources/ButterSwift/Models/
# Text/Gemma4Text.swift's `Gemma4Loader.load` / `buildMoE` / `sliceStacked`
# (this repo, branch w43/qwen-mtp-loop @ 9f406b7a). No `quantization` key
# is written, so every `loadLinear` call takes the plain (unquantized)
# `Linear` branch — F16 dense weights throughout. This sidesteps having to
# hand-roll MLX's packed affine-int4 bitstream (the real checkpoint's
# format) for what is purely an allocation-volume experiment: the
# `Ops.dequantGemv` kernels write only an activation-sized output buffer
# (verified by reading IronSwift/Generated/IronKernels.swift), so plain
# vs. quantized weights should not change the alloc-byte totals we're
# tracing at all.
#
# Usage:
#   python3 scratch/make_gemma4_moe_synthetic_checkpoint.py \
#       --out-dir /tmp/gemma4-moe-synthetic \
#       --hidden 256 --num-layers 2 --num-experts 4 --top-k 2 \
#       --moe-intermediate 512
#
# Re-run with different --num-layers / --num-experts / --hidden /
# --moe-intermediate to build the scaling-law data points (see
# Tests/ModelIntegrationTests/Text/Gemma4MoEAllocTraceTests.swift, which
# drives this script's output through BUTTER_ALLOC_TRACE=1).

import argparse
import json
import math
import os
import random
import struct

# Pure-stdlib on purpose: this generator runs inside the CI test job,
# whose python has no numpy. A "tensor" is just (shape, dtype_tag,
# raw_bytes) — allocation-behavior tests never care about the values.


class T:
    __slots__ = ("shape", "dtype", "data")

    def __init__(self, shape, dtype, data):
        self.shape = list(shape)
        self.dtype = dtype  # "F16" | "F32" | "U32"
        self.data = data


def _nelem(shape) -> int:
    return math.prod(shape) if shape else 1


def save_safetensors(tensors: dict, path: str) -> None:
    """Write `tensors` (name -> T) as a safetensors v2 file.

    Mirrors the format Sources/ButterSwift/Loader/SafeTensors.swift reads:
    8-byte LE header length, UTF-8 JSON header, then raw tensor bytes
    concatenated in header order. No alignment padding — the Swift reader
    mmaps the whole file and indexes by absolute byte offset, so
    unaligned data_offsets are fine.
    """
    header = {}
    blobs = []
    offset = 0
    for name, t in tensors.items():
        nbytes = len(t.data)
        header[name] = {
            "dtype": t.dtype,
            "shape": t.shape,
            "data_offsets": [offset, offset + nbytes],
        }
        blobs.append(t.data)
        offset += nbytes
    header_json = json.dumps(header, sort_keys=True).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for b in blobs:
            f.write(b)


def _pack_f16(values) -> bytes:
    return struct.pack(f"<{len(values)}e", *values)


def rand_f16(rng: random.Random, shape, std: float = 0.02) -> T:
    n = _nelem(shape)
    return T(shape, "F16", _pack_f16([rng.gauss(0.0, std) for _ in range(n)]))


def norm_f16(rng: random.Random, shape, center: float = 1.0, spread: float = 0.1) -> T:
    n = _nelem(shape)
    return T(
        shape, "F16",
        _pack_f16([center + rng.uniform(-spread, spread) for _ in range(n)]))


def build(args) -> None:
    rng = random.Random(args.seed)
    hidden = args.hidden
    num_layers = args.num_layers
    num_experts = args.num_experts
    top_k = args.top_k
    moe_intermediate = args.moe_intermediate
    intermediate = args.intermediate or moe_intermediate
    n_heads = args.num_heads
    n_kv_heads = args.num_kv_heads
    n_global_kv_heads = args.num_global_kv_heads
    head_dim = args.head_dim
    global_head_dim = args.global_head_dim
    vocab = args.vocab

    assert hidden % 128 == 0, "hidden must be 128-aligned (Gemma4Loader.load norm-width check)"
    assert head_dim % 128 == 0 and head_dim in (128, 256, 512), "head_dim must be one of 128/256/512"
    assert global_head_dim % 128 == 0 and global_head_dim in (128, 256, 512), (
        "global_head_dim must be one of 128/256/512"
    )
    assert top_k <= num_experts

    os.makedirs(args.out_dir, exist_ok=True)

    # layer_types: every 6th layer (1-indexed) is full_attention, matching
    # the real 26B-A4B checkpoint's 5-full-of-30 ratio.
    layer_types = [
        "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
        for i in range(num_layers)
    ]
    if num_layers < 6:
        # Too few layers for the real ratio to show up — force at least
        # one of each so both attention geometries (and both FFN
        # command-buffer paths) get exercised.
        layer_types[0] = "sliding_attention"
        layer_types[-1] = "full_attention" if num_layers > 1 else layer_types[0]

    config = {
        "architectures": ["Gemma4TextForCausalLM"],
        "model_type": "gemma4_text",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "num_global_key_value_heads": n_global_kv_heads,
        "head_dim": head_dim,
        "global_head_dim": global_head_dim,
        "intermediate_size": intermediate,
        "moe_intermediate_size": moe_intermediate,
        "num_experts": num_experts,
        "top_k_experts": top_k,
        "rms_norm_eps": 1e-6,
        "vocab_size": vocab,
        "max_position_embeddings": args.max_position_embeddings,
        "sliding_window": args.sliding_window,
        "layer_types": layer_types,
        "rope_parameters": {
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            "full_attention": {
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
                "rope_type": "proportional",
            },
        },
        "final_logit_softcapping": 30.0,
        "tie_word_embeddings": True,
        "attention_k_eq_v": False,
        "use_double_wide_mlp": False,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "vocab_size_per_layer_input": vocab,
        "enable_moe_block": True,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    tensors = {}
    tensors["model.embed_tokens.weight"] = rand_f16(rng, (vocab, hidden))

    for i in range(num_layers):
        lp = f"model.layers.{i}"
        is_global = layer_types[i] == "full_attention"
        layer_head_dim = global_head_dim if is_global else head_dim
        layer_kv_heads = n_global_kv_heads if is_global else n_kv_heads

        tensors[f"{lp}.self_attn.q_proj.weight"] = rand_f16(rng, (n_heads * layer_head_dim, hidden))
        tensors[f"{lp}.self_attn.k_proj.weight"] = rand_f16(
            rng, (layer_kv_heads * layer_head_dim, hidden)
        )
        tensors[f"{lp}.self_attn.v_proj.weight"] = rand_f16(
            rng, (layer_kv_heads * layer_head_dim, hidden)
        )
        tensors[f"{lp}.self_attn.o_proj.weight"] = rand_f16(rng, (hidden, n_heads * layer_head_dim))
        tensors[f"{lp}.self_attn.q_norm.weight"] = norm_f16(rng, (layer_head_dim,))
        tensors[f"{lp}.self_attn.k_norm.weight"] = norm_f16(rng, (layer_head_dim,))

        tensors[f"{lp}.input_layernorm.weight"] = norm_f16(rng, (hidden,))
        tensors[f"{lp}.post_attention_layernorm.weight"] = norm_f16(rng, (hidden,))
        tensors[f"{lp}.pre_feedforward_layernorm.weight"] = norm_f16(rng, (hidden,))
        tensors[f"{lp}.post_feedforward_layernorm.weight"] = norm_f16(rng, (hidden,))

        # Shared dense GELU MLP branch (h1).
        tensors[f"{lp}.mlp.gate_proj.weight"] = rand_f16(rng, (intermediate, hidden))
        tensors[f"{lp}.mlp.up_proj.weight"] = rand_f16(rng, (intermediate, hidden))
        tensors[f"{lp}.mlp.down_proj.weight"] = rand_f16(rng, (hidden, intermediate))

        # Router.
        tensors[f"{lp}.router.proj.weight"] = rand_f16(rng, (num_experts, hidden))
        tensors[f"{lp}.router.scale"] = norm_f16(rng, (hidden,))
        tensors[f"{lp}.router.per_expert_scale"] = norm_f16(rng, (num_experts,), center=1.0, spread=0.05)

        # Stacked experts — raw (unquantized) `[numExperts, outDim, inDim]`.
        tensors[f"{lp}.experts.switch_glu.gate_proj.weight"] = rand_f16(
            rng, (num_experts, moe_intermediate, hidden)
        )
        tensors[f"{lp}.experts.switch_glu.up_proj.weight"] = rand_f16(
            rng, (num_experts, moe_intermediate, hidden)
        )
        tensors[f"{lp}.experts.switch_glu.down_proj.weight"] = rand_f16(
            rng, (num_experts, hidden, moe_intermediate)
        )

        # MoE block's three extra norms.
        tensors[f"{lp}.pre_feedforward_layernorm_2.weight"] = norm_f16(rng, (hidden,))
        tensors[f"{lp}.post_feedforward_layernorm_1.weight"] = norm_f16(rng, (hidden,))
        tensors[f"{lp}.post_feedforward_layernorm_2.weight"] = norm_f16(rng, (hidden,))

        # Per-layer output scalar [1] — set to exactly 1.0 so
        # `applyPLEAndScalar` skips the extra multiply (its `scalar !=
        # 1.0` fast path), keeping the synthetic forward's op count a
        # clean match for the naive-MoE-loop math this experiment cares
        # about.
        tensors[f"{lp}.layer_scalar"] = T([1], "F16", _pack_f16([1.0]))

    tensors["model.norm.weight"] = norm_f16(rng, (hidden,))
    # tie_word_embeddings=True: no separate lm_head.weight tensor.

    save_safetensors(tensors, os.path.join(args.out_dir, "model.safetensors"))

    total_bytes = sum(len(t.data) for t in tensors.values())
    print(
        f"wrote {len(tensors)} tensors, {total_bytes / 1e6:.2f} MB, to {args.out_dir}\n"
        f"  hidden={hidden} num_layers={num_layers} num_experts={num_experts} "
        f"top_k={top_k} moe_intermediate={moe_intermediate} intermediate={intermediate}\n"
        f"  head_dim={head_dim} global_head_dim={global_head_dim} vocab={vocab} "
        f"max_position_embeddings={args.max_position_embeddings}"
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--moe-intermediate", type=int, default=512)
    p.add_argument(
        "--intermediate", type=int, default=None,
        help="Shared-MLP intermediate size. Defaults to --moe-intermediate.",
    )
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--num-kv-heads", type=int, default=1)
    p.add_argument("--num-global-kv-heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--global-head-dim", type=int, default=128)
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument("--max-position-embeddings", type=int, default=4096)
    p.add_argument("--sliding-window", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())

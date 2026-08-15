#!/usr/bin/env python3
# Offline requant: MLX 4-bit affine g64 lm_head -> MLX 3-bit affine g64
# lm_head sidecar, for butter's MTP draft-only lm_head experiment
# (BUTTER_QWEN_MTP_DRAFT_LMHEAD_3BIT=1). See Qwen3xMTPHead.swift.
#
# Reads ONLY the lm_head.{weight,scales,biases} byte ranges out of the
# sharded safetensors checkpoint (never loads the whole ~14GB model),
# dequantizes to bf16 in row-chunks (bounds memory), requantizes each
# chunk to 3-bit/group_size=64 via mlx.core.quantize (the same affine
# packing mlx-community's published 3-bit checkpoints use — this is
# exactly what Butter's `dequant_gemv_int3` kernel is validated against
# in Quantized3bitIntegrationTests, so round-tripping through mx.quantize
# guarantees layout compatibility instead of hand-rolling the bitstream
# pack).
#
# Output: a single safetensors file next to the model,
# `draft_lm_head_3bit.safetensors`, with three top-level tensors:
#   weight   [vocab, in_dim*3/32]  u32   (packed, MLX affine layout)
#   scales   [vocab, in_dim/64]    bf16
#   biases   [vocab, in_dim/64]    bf16
#
# Usage:
#   python3 scratch/quantize_draft_lmhead_3bit.py \
#       --model-dir /Users/tom/models/qwen36-27b-4bit-mlx

import argparse
import json
import mmap
import os
import struct
import sys

import mlx.core as mx
import numpy as np


TENSOR_KEY = "language_model.lm_head"
OUT_BITS = 3
GROUP_SIZE = 64


def read_shard_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    data_start = 8 + hlen
    return header, data_start


def find_lm_head_shard(model_dir):
    """Resolve which shard file carries the lm_head triplet + the byte
    offsets/shapes/dtypes for weight/scales/biases, without touching any
    other tensor in the checkpoint."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    names = {
        "weight": f"{TENSOR_KEY}.weight",
        "scales": f"{TENSOR_KEY}.scales",
        "biases": f"{TENSOR_KEY}.biases",
    }
    shard_files = {names[k]: weight_map[names[k]] for k in names}
    # All three should live in the same shard for this checkpoint, but
    # don't assume it — resolve independently.
    entries = {}
    headers_by_file = {}
    for k, tname in names.items():
        shard = shard_files[tname]
        shard_path = os.path.join(model_dir, shard)
        if shard_path not in headers_by_file:
            headers_by_file[shard_path] = read_shard_header(shard_path)
        header, data_start = headers_by_file[shard_path]
        meta = header[tname]
        entries[k] = {
            "path": shard_path,
            "data_start": data_start,
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "offsets": meta["data_offsets"],
        }
    return entries


DTYPE_ITEMSIZE = {"U32": 4, "BF16": 2}


def read_rows(mm, entry, row0, row1):
    """Read rows [row0, row1) of a 2D row-major tensor straight out of
    the mmap'd shard, as a numpy array of raw dtype (uint32 or uint16
    for bf16-as-bits)."""
    rows, cols = entry["shape"]
    itemsize = DTYPE_ITEMSIZE[entry["dtype"]]
    row_stride = cols * itemsize
    base = entry["data_start"] + entry["offsets"][0]
    start = base + row0 * row_stride
    length = (row1 - row0) * row_stride
    raw = mm[start : start + length]
    np_dtype = np.uint32 if entry["dtype"] == "U32" else np.uint16
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(row1 - row0, cols)
    return arr


def mx_from_u32(np_arr):
    return mx.array(np_arr)


def mx_bf16_from_bits(np_u16_arr):
    return mx.array(np_u16_arr).view(mx.bfloat16)


def write_safetensors(path, tensors, metadata):
    """Minimal safetensors writer: tensors is {name: (mx_array, dtype_str)}."""
    header = {}
    blobs = []
    offset = 0
    for name, (arr, dtype_str) in tensors.items():
        # mx arrays -> numpy -> bytes. bf16 arrays need a bit-view to
        # uint16 first (numpy has no bf16 dtype).
        if dtype_str == "BF16":
            npv = np.array(arr.view(mx.uint16))
        else:
            npv = np.array(arr)
        raw = npv.tobytes()
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    header["__metadata__"] = metadata
    header_json = json.dumps(header).encode("utf-8")
    # safetensors requires 8-byte-aligned header length in some readers;
    # pad with spaces to a multiple of 8 for safety.
    pad = (-len(header_json)) % 8
    header_json += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir", default="/Users/tom/models/qwen36-27b-4bit-mlx"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk-rows", type=int, default=8192)
    ap.add_argument(
        "--spot-check-rows",
        type=int,
        default=8,
        help="Rows to spot-check (4-bit vs 3-bit dequant delta) after conversion.",
    )
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        args.model_dir, "draft_lm_head_3bit.safetensors"
    )

    entries = find_lm_head_shard(args.model_dir)
    w_entry, s_entry, b_entry = entries["weight"], entries["scales"], entries["biases"]
    vocab, packed_cols = w_entry["shape"]
    _, n_groups = s_entry["shape"]
    in_dim = n_groups * GROUP_SIZE
    assert packed_cols * 32 // 4 == in_dim, (
        f"unexpected 4-bit packing: packed_cols={packed_cols} in_dim={in_dim}"
    )
    print(
        f"[quantize_draft_lmhead_3bit] source lm_head: vocab={vocab} "
        f"in_dim={in_dim} groups={n_groups} (4-bit g{GROUP_SIZE})",
        file=sys.stderr,
    )

    out_packed_cols = in_dim * OUT_BITS // 32
    print(
        f"[quantize_draft_lmhead_3bit] target: 3-bit g{GROUP_SIZE}, "
        f"packed_cols={out_packed_cols}",
        file=sys.stderr,
    )

    shard_path = w_entry["path"]
    assert s_entry["path"] == shard_path and b_entry["path"] == shard_path, (
        "lm_head triplet split across shards — read_rows below assumes one file"
    )

    fd = os.open(shard_path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    finally:
        os.close(fd)

    weight_chunks = []
    scales_chunks = []
    biases_chunks = []

    max_abs_err = 0.0
    sum_abs_err = 0.0
    sum_abs_ref = 0.0
    n_checked = 0

    row = 0
    while row < vocab:
        row_end = min(row + args.chunk_rows, vocab)

        w_np = read_rows(mm, w_entry, row, row_end)  # u32 [rows, packed_cols]
        s_np = read_rows(mm, s_entry, row, row_end)  # u16-bits [rows, n_groups]
        b_np = read_rows(mm, b_entry, row, row_end)

        w4 = mx_from_u32(w_np)
        s4 = mx_bf16_from_bits(s_np)
        b4 = mx_bf16_from_bits(b_np)

        dequant = mx.dequantize(w4, s4, b4, group_size=GROUP_SIZE, bits=4)
        w3, s3, b3 = mx.quantize(dequant, group_size=GROUP_SIZE, bits=OUT_BITS)
        mx.eval(w3, s3, b3)

        weight_chunks.append(np.array(w3))
        scales_chunks.append(np.array(s3.view(mx.uint16)))
        biases_chunks.append(np.array(b3.view(mx.uint16)))

        # Spot-check the first `spot_check_rows` of THIS chunk only, once
        # per run (row==0), against a re-dequant through the 3-bit params.
        if row == 0 and args.spot_check_rows > 0:
            k = min(args.spot_check_rows, row_end - row)
            dequant3 = mx.dequantize(
                w3[:k], s3[:k], b3[:k], group_size=GROUP_SIZE, bits=OUT_BITS
            )
            ref = dequant[:k].astype(mx.float32)
            got = dequant3.astype(mx.float32)
            err = mx.abs(got - ref)
            mx.eval(err)
            max_abs_err = max(max_abs_err, float(err.max()))
            sum_abs_err += float(err.sum())
            sum_abs_ref += float(mx.abs(ref).sum())
            n_checked += k * in_dim

        row = row_end
        print(
            f"[quantize_draft_lmhead_3bit] rows {row}/{vocab}", file=sys.stderr
        )

    del mm

    weight_full = mx.array(np.concatenate(weight_chunks, axis=0))
    scales_full_bits = np.concatenate(scales_chunks, axis=0)
    biases_full_bits = np.concatenate(biases_chunks, axis=0)
    scales_full = mx.array(scales_full_bits).view(mx.bfloat16)
    biases_full = mx.array(biases_full_bits).view(mx.bfloat16)

    assert weight_full.shape == (vocab, out_packed_cols)
    assert scales_full.shape == (vocab, n_groups)
    assert biases_full.shape == (vocab, n_groups)

    write_safetensors(
        out_path,
        {
            "weight": (weight_full, "U32"),
            "scales": (scales_full, "BF16"),
            "biases": (biases_full, "BF16"),
        },
        metadata={
            "format": "mlx-affine-quantized-draft-lm-head",
            "source": "language_model.lm_head (4-bit g64 affine)",
            "bits": str(OUT_BITS),
            "group_size": str(GROUP_SIZE),
            "vocab": str(vocab),
            "in_dim": str(in_dim),
        },
    )

    out_size = os.path.getsize(out_path)
    print(
        f"[quantize_draft_lmhead_3bit] wrote {out_path} "
        f"({out_size / 1e9:.3f} GB)",
        file=sys.stderr,
    )
    if n_checked > 0:
        print(
            f"[quantize_draft_lmhead_3bit] spot-check ({n_checked} elements, "
            f"first {args.spot_check_rows} rows): "
            f"max_abs_err={max_abs_err:.5f} "
            f"mean_abs_err={sum_abs_err / n_checked:.5f} "
            f"rel_l1={sum_abs_err / max(sum_abs_ref, 1e-9):.4f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

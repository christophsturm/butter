#!/usr/bin/env bash
# Apply butter/laguna kernel overlays and build kernels_extra.metallib.
#
# iron build --emit writes kernels.metallib from iron-only airs. Full re-link
# against a foreign iron-dev air cache can silently regress other kernels, so
# we keep the iron metallib intact and ship a small overlay metallib:
#
#   Sources/IronSwift/Resources/kernels_extra.metallib
#
# IronLibrary loads it best-effort; PSOCache prefers extra over main on name
# clash (fast NVFP4 eidx qdot) and for laguna Path B fuse kernels.
#
# Usage: from butter repo root, `make link-extra-metals` (also hooked from
# regenerate-kernels). Avoids `swift package clean`.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RES="$ROOT/Sources/IronSwift/Resources"
KDIR="$RES/kernels"
OVERLAY="$RES/kernel_overlays"
EXTRA_METALLIB="$RES/kernels_extra.metallib"
AIR_DIR="${IRON_AIR_DIR:-$ROOT/.build/tile-build-air-extra}"
SDK="${METAL_SDK:-macosx}"
mkdir -p "$KDIR" "$AIR_DIR"

if [[ ! -d "$OVERLAY" ]]; then
  echo "link-extra-metals: no overlays at $OVERLAY (ok — nothing to do)"
  exit 0
fi

# 1) Copy overlays into kernels/ (source available for live-compile fallback).
count=0
for src in "$OVERLAY"/*.metal; do
  [[ -f "$src" ]] || continue
  base="$(basename "$src")"
  cp "$src" "$KDIR/$base"
  count=$((count + 1))
done
echo "link-extra-metals: applied $count overlay(s) → $KDIR"

# 2) Compile overlay metals → air (only overlays; not full butter_ MPP set —
#    those stay live-compile-from-source via PSOCache isMppKernel).
airs=()
for src in "$OVERLAY"/*.metal; do
  [[ -f "$src" ]] || continue
  base="$(basename "$src" .metal)"
  air="$AIR_DIR/${base}.air"
  echo "  metal → air: $base"
  xcrun -sdk "$SDK" metal -c "$src" -o "$air"
  airs+=("$air")
done

if [[ ${#airs[@]} -eq 0 ]]; then
  echo "link-extra-metals: no airs to link"
  exit 0
fi

echo "link-extra-metals: linking ${#airs[@]} airs → $EXTRA_METALLIB"
xcrun -sdk "$SDK" metallib "${airs[@]}" -o "$EXTRA_METALLIB"
ls -la "$EXTRA_METALLIB"

for n in laguna_moe_routed_swiglu_qmv_nvfp4_bf16 \
         laguna_moe_routed_down_reduce_nvfp4_bf16 \
         nemotron_moe_routed_relu2_qmv_nvfp4_bf16 \
         nemotron_moe_routed_down_reduce_nvfp4_bf16 \
         iron_nvfp4_dequant_gemv_expert_indexed_bf16 \
         iron_qmm_coop_bm64_bf16; do
  if xcrun metal-nm "$EXTRA_METALLIB" 2>/dev/null | grep -q " $n\$"; then
    echo "  ok: $n in kernels_extra.metallib"
  else
    echo "  WARN: $n not found in kernels_extra.metallib" >&2
  fi
done
echo "link-extra-metals: done"

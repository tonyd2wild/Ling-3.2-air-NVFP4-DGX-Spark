#!/usr/bin/env bash
# Builds the two patched vLLM files that ../launch_fp8.sh mounts read-only over the image copies.
# 1. copies the stock files out of the pinned vLLM image (docker create + docker cp, nothing runs),
# 2. checks their sha256 against the image we tested,
# 3. applies the .patch files next to this script and checks the result.
# Output: fp8kv/patched/triton_decode_attention.py and fp8kv/patched/extensible_kv_cache.py
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-eff4de9be3c901c6ad12cb651de64e4b10ced219}"
SITE=/usr/local/lib/python3.14/dist-packages
# path inside site-packages | sha256 of the stock file | sha256 after the patch
FILES=(
  "vllm/v1/attention/ops/triton_decode_attention.py cfcceec1d5dc64bca60474fb237f678e1634f7d238a029c20354f11b00ffe420 125242fa5c2fac6de525fccf99369c976f78888a22837d930aac815f9b85227f"
  "vllm/v1/worker/extensible_kv_cache.py 9769fda3c812896f10515deacf0f45f62a62fc6e48c6acddf868d14e43cf634a 8992be16fdf8b8cd5569a23686112a7cc34775136ff2783490155d9a48f1385d"
)
command -v patch >/dev/null || { echo "the 'patch' tool is missing (sudo apt install patch)" >&2; exit 1; }
mkdir -p "$HERE/stock" "$HERE/patched"
cid="$(docker create "$IMAGE")"
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
for row in "${FILES[@]}"; do
  read -r rel want_stock want_patched <<<"$row"
  base="$(basename "$rel")"
  docker cp "$cid:$SITE/$rel" "$HERE/stock/$base"
  got="$(sha256sum "$HERE/stock/$base" | cut -d' ' -f1)"
  if [ "$got" != "$want_stock" ]; then
    echo "stock $rel in $IMAGE is not the version these patches were made for (sha256 $got)." >&2
    echo "Use the pinned image, or apply the patch by hand and review it." >&2
    exit 1
  fi
  patch -s -o "$HERE/patched/$base" "$HERE/stock/$base" "$HERE/${base%.py}.patch"
  chmod 644 "$HERE/patched/$base"
  got="$(sha256sum "$HERE/patched/$base" | cut -d' ' -f1)"
  [ "$got" = "$want_patched" ] || { echo "patched $base has an unexpected sha256 ($got)" >&2; exit 1; }
  echo "ok  fp8kv/patched/$base"
done
echo "done. Start the server with ./launch_fp8.sh"

#!/usr/bin/env bash
# Ling-3.2-air (NVFP4) on one DGX Spark (GB10): vLLM nightly, 262K context, MTP on (3 draft tokens), FP8 KV cache.
# Same as launch.sh plus:
#   --kv-cache-dtype fp8      about 40% less memory per cached token -> 2.17M-token pool at UTIL 0.765
#   two patched vLLM files, mounted read-only over the image copies (build them once with ./fp8kv/prepare.sh):
#     triton_decode_attention.py  the Triton MLA decode kernel needs 102,400 B of shared memory for an FP8 KV
#                                 cache at num_stages=2; GB10 allows 101,376 B. The patch uses one stage there.
#     extensible_kv_cache.py      optional cap (FP8FIX_TORCH_GROWTH_GIB) on how far the torch caching allocator
#                                 may grow after warmup, so a long prefill cannot eat the OS headroom.
# Usage:   ./fp8kv/prepare.sh && ./launch_fp8.sh
#          SPEC= ./launch_fp8.sh           # MTP off
#          TORCH_GROWTH_GIB= ./launch_fp8.sh   # no growth cap (stock behavior)
# Extra vLLM flags can be appended: ./launch_fp8.sh --max-num-seqs 4
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${NAME:-ling32air}"
PORT="${PORT:-8140}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Ling-3.2-air-NVFP4}"
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-eff4de9be3c901c6ad12cb651de64e4b10ced219}"
MAXLEN="${MAXLEN:-262144}"
UTIL="${UTIL:-0.765}"
RESTART="${RESTART:-unless-stopped}"
TORCH_GROWTH_GIB="${TORCH_GROWTH_GIB-10}"
SPEC="${SPEC-{\"method\":\"mtp\",\"num_speculative_tokens\":3}}"
SITE=/usr/local/lib/python3.14/dist-packages
P1="$HERE/fp8kv/patched/triton_decode_attention.py"
P2="$HERE/fp8kv/patched/extensible_kv_cache.py"

[ -f "$MODEL_DIR/config.json" ] || { echo "weights not found in $MODEL_DIR (see README section 3)"; exit 1; }
[ -f "$P1" ] && [ -f "$P2" ] || { echo "patched files missing: run ./fp8kv/prepare.sh first"; exit 1; }

SPEC_ARGS=()
[ -n "$SPEC" ] && SPEC_ARGS=(--speculative-config "$SPEC")
ENV_ARGS=()
[ -n "$TORCH_GROWTH_GIB" ] && ENV_ARGS=(-e "FP8FIX_TORCH_GROWTH_GIB=$TORCH_GROWTH_GIB")

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host --restart "$RESTART" -p "$PORT:8000" \
  -v "$MODEL_DIR:/models/Ling-3.2-air-NVFP4:ro" \
  -v "$P1:$SITE/vllm/v1/attention/ops/triton_decode_attention.py:ro" \
  -v "$P2:$SITE/vllm/v1/worker/extensible_kv_cache.py:ro" \
  "${ENV_ARGS[@]}" "$IMAGE" \
  --model /models/Ling-3.2-air-NVFP4 \
  --served-model-name ling-3.2-air primitive-ai/Ling-3.2-air-NVFP4 \
  --trust-remote-code \
  --max-model-len "$MAXLEN" \
  --max-num-seqs 8 \
  --gpu-memory-utilization "$UTIL" \
  --kv-cache-dtype fp8 \
  --reasoning-parser ling3 \
  --enable-auto-tool-choice --tool-call-parser ling3 \
  "${SPEC_ARGS[@]}" "$@"

echo "started $NAME on :$PORT (maxlen $MAXLEN, util $UTIL, kv fp8, spec ${SPEC:-off}, torch growth cap ${TORCH_GROWTH_GIB:-off} GiB)"
echo "first start takes about 4 minutes. Follow it with: docker logs -f $NAME   (ready at 'Application startup complete')"
echo "check the log for: GPU KV cache size: 2,170,119 tokens   and   [fp8fix] capping torch reservation"

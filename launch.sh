#!/usr/bin/env bash
# Ling-3.2-air (NVFP4) on one DGX Spark (GB10): vLLM nightly, 262K context, MTP on (3 draft tokens).
# Usage:   ./launch.sh                 # MTP on (default)
#          SPEC= ./launch.sh           # MTP off
#          PORT=8000 ./launch.sh       # another host port
# Extra vLLM flags can be appended: ./launch.sh --max-num-seqs 4
set -euo pipefail
NAME="${NAME:-ling32air}"
PORT="${PORT:-8140}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Ling-3.2-air-NVFP4}"
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-eff4de9be3c901c6ad12cb651de64e4b10ced219}"
MAXLEN="${MAXLEN:-262144}"
UTIL="${UTIL:-0.80}"
RESTART="${RESTART:-unless-stopped}"
SPEC="${SPEC-{\"method\":\"mtp\",\"num_speculative_tokens\":3}}"

[ -f "$MODEL_DIR/config.json" ] || { echo "weights not found in $MODEL_DIR (see README section 3)"; exit 1; }

SPEC_ARGS=()
[ -n "$SPEC" ] && SPEC_ARGS=(--speculative-config "$SPEC")

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host --restart "$RESTART" -p "$PORT:8000" \
  -v "$MODEL_DIR:/models/Ling-3.2-air-NVFP4:ro" "$IMAGE" \
  --model /models/Ling-3.2-air-NVFP4 \
  --served-model-name ling-3.2-air primitive-ai/Ling-3.2-air-NVFP4 \
  --trust-remote-code \
  --max-model-len "$MAXLEN" \
  --max-num-seqs 8 \
  --gpu-memory-utilization "$UTIL" \
  --reasoning-parser ling3 \
  --enable-auto-tool-choice --tool-call-parser ling3 \
  "${SPEC_ARGS[@]}" "$@"

echo "started $NAME on :$PORT (maxlen $MAXLEN, util $UTIL, spec ${SPEC:-off})"
echo "first start takes about 4 minutes (weights ~75 s, compile + graphs ~2.5 min)."
echo "follow it with: docker logs -f $NAME   (ready at 'Application startup complete')"

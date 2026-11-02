#!/usr/bin/env bash
# Kinetick-labs Spark prod: Qwen3.8 Flash Next FP4-Local on TensorFold's mia engine (flashnext), :8083.
# fp8 KV, 12 GiB prompt cache, 3 lanes x 262,144 tokens, vision helper, DFlash2 drafts,
# strict tools / structured output (grammar hooks after Jay Leaton's upstream strict-tool work),
# layer-sliced prefill (TF_FLASHNEXT_FILL, after MiaAI-Lab) with one long prompt at a time (srf),
# one-layer slices and 1024-row chunks while another stream decodes.
#
# Layout (stable, not ad hoc): $ROOT/bin/tensorfold-native (ReleaseFast build of the pushed commit),
# $ROOT/libs (libtfgrammar.so, libturbojpeg, libwebpdecoder), $ROOT/COMMIT.
# Usage: serve.sh            start (replaces the container)   | DRY_RUN=1 serve.sh  print the command
set -euo pipefail
ROOT=${ROOT:-$HOME/tf-prod-df2g}
NAME=${CONTAINER_NAME:-qwen38-fp4local}
PORT=${PORT:-8083}
IMAGE=${IMAGE:-tensorfold-zig:exl3-572414d}          # runtime only (CUDA libs); the binary comes from $ROOT/bin
SNAP=/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/${MODEL_REVISION:-b9ce18a6887e18c089fdea8509a33e63429a3300}
DFLASH=${DFLASH:-$HOME/models/qwen38-dflash2}
VISION_W=${VISION_W:-$HOME/models/sxuff-vision-bf16}
for f in "$ROOT/bin/tensorfold-native" "$ROOT/libs/libtfgrammar.so" "$DFLASH" "$VISION_W/vision_tower_bf16.safetensors"; do
  [[ -e "$f" ]] || { echo "serve.sh: missing $f" >&2; exit 1; }
done
ENV=(
  TENSORFOLD_NO_UPDATE_CHECK=1 HF_HUB_OFFLINE=1 LD_LIBRARY_PATH=/tfp/libs
  TENSORFOLD_FLASHNEXT_ENGINE=mia TENSORFOLD_VISION_WEIGHTS=/vis/vision_tower_bf16.safetensors
  TF_FLASHNEXT_PREFILL_ROWS=4096 TF_FLASHNEXT_DEPTH=15 TF_FLASHNEXT_CONFIDENCE=-0.4 TF_FLASHNEXT_PRODUCT_STREAMS=2
  # DFlash2: 7 drafts, cut 0.15, wide cut 0.5
  TF_FLASHNEXT_DFLASH=/dflash TF_FLASHNEXT_DFLASH_DRAFTS=7 TF_FLASHNEXT_DFLASH_CUT=0.15 TF_FLASHNEXT_DFLASH_CUT_WIDE=0.5
  # prefill beside decoders: sliced (MiaAI-Lab TF_FLASHNEXT_FILL), one long prompt at a time, 1 layer and
  # 1024-row chunks while another stream decodes (held 3 s after the last decode round, the default)
  TF_FLASHNEXT_FILL=1 TENSORFOLD_PREFILL_POLICY=srf TF_FLASHNEXT_FILL_LAYERS_BUSY=1 TF_FLASHNEXT_FILL_ROWS_BUSY=1024
)
E=(); for e in "${ENV[@]}"; do E+=(-e "$e"); done
CMD=(docker run -d --name "$NAME" --gpus all --ipc=host --network host
     --ulimit memlock=-1 --ulimit stack=67108864 "${E[@]}"
     -v "$ROOT":/tfp:ro -v "$HOME/.cache/huggingface":/root/.cache/huggingface:ro
     -v "$VISION_W":/vis:ro -v "$DFLASH":/dflash:ro
     --entrypoint /tfp/bin/tensorfold-native "$IMAGE"
     serve "$SNAP" --name Qwen3.8-Flash-Next --host 0.0.0.0 --port "$PORT"
     --context 262144 --parallel 3 --max-tokens 32768 --temperature 1.0 --top-p 0.95 --top-k 20 --thinking
     --kv-cache fp8 --prompt-cache-gib 12 --vision-helper)
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
"${CMD[@]}" >/dev/null
for i in $(seq 200); do
  curl -sf -m3 "http://127.0.0.1:$PORT/health" >/dev/null && { echo "serve.sh: $NAME ready on :$PORT ($(cat "$ROOT/COMMIT" 2>/dev/null))"; exit 0; }
  [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" == true ]] || { docker logs --tail 40 "$NAME" >&2; exit 1; }
  sleep 3
done
echo "serve.sh: not ready after 600 s" >&2; exit 1

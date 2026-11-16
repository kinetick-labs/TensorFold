#!/usr/bin/env bash
# Kinetick-labs Spark: GLM-5.3-Flash SAGE REXL3 2.48bpw on TensorFold (feat/glm53-rexl3), :8094.
# 2 streams x 262,144-token windows, paged KV (TF_KV_PAGES=on), 5 GiB memory reserve,
# vision, DFlash2 drafts, conversation reuse (TF_KV_SHARE=2 kept states, their KDA snapshots in
# host RAM: TF_GLM_HOST_CACHE=1, --prompt-cache-gib 1), short prompts first between 2048-row
# prompt passes (TENSORFOLD_PREFILL_POLICY=fcfs). Not prod (Qwen on :8083 is unchanged).
#
# Layout: $BIN/bin/tensorfold-native (ReleaseFast build of this branch), $LIBS (CUDA/vision
# runtime libs). Usage: serve.sh            start (replaces the glm53 container)
#                     DRY_RUN=1 serve.sh  print the command
set -euo pipefail
ROOT=${ROOT:-$HOME/tf-glm53}
if [[ -z "${BIN:-}" ]]; then
  if [[ -x "$ROOT/zig-out/native/bin/tensorfold-native" ]]; then
    BIN=$ROOT/zig-out/native
  else
    BIN=$ROOT
  fi
fi
LIBS=${LIBS:-$HOME/tf-prod-df2g/libs}
NAME=${CONTAINER_NAME:-glm53}
PORT=${PORT:-8094}
IMAGE=${IMAGE:-tensorfold-zig:exl3-572414d}
MODEL=${MODEL:-$HOME/models/GLM-5.3-Flash-SAGE-REXL3-2.48bpw}
DFLASH=${DFLASH:-$HOME/models/glm53-dflash2}
for f in "$BIN/bin/tensorfold-native" "$LIBS/libtfgrammar.so" "$MODEL" "$DFLASH"; do
  [[ -e "$f" ]] || { echo "serve.sh: missing $f" >&2; exit 1; }
done
ENV=(
  TENSORFOLD_NO_UPDATE_CHECK=1 HF_HUB_OFFLINE=1 LD_LIBRARY_PATH=/tfl
  TF_GLM_RANK=0 TF_GLM_WORLD=1
  TF_KV_PAGES=on TENSORFOLD_MEMORY_RESERVE_GIB=5
  TF_KV_SHARE=2 TF_GLM_HOST_CACHE=1 TENSORFOLD_PREFILL_POLICY=fcfs
)
E=(); for e in "${ENV[@]}"; do E+=(-e "$e"); done
CMD=(docker run -d --name "$NAME" --gpus all --ipc=host --network host
     --ulimit memlock=-1 --ulimit stack=67108864 "${E[@]}"
     -v "$BIN":/tfb:ro -v "$LIBS":/tfl:ro -v "$MODEL":/model:ro -v "$DFLASH":/dflash:ro
     -v "$HOME/.cache/tensorfold":/root/.cache/tensorfold
     --entrypoint /tfb/bin/tensorfold-native "$IMAGE"
     serve /model --name GLM-5.3-Flash --host 0.0.0.0 --port "$PORT"
     --context 262144 --parallel 2 --max-tokens 8192 --prompt-cache-gib 1
     --vision --drafter /dflash)
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
"${CMD[@]}" >/dev/null
for i in $(seq 200); do
  curl -sf -m3 "http://127.0.0.1:$PORT/health" >/dev/null && {
    echo "serve.sh: $NAME ready on :$PORT ($(git -C "$ROOT" rev-parse --short=9 HEAD 2>/dev/null || cat "$BIN/COMMIT" 2>/dev/null || echo unknown))"
    exit 0
  }
  [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" == true ]] || { docker logs --tail 40 "$NAME" >&2; exit 1; }
  sleep 3
done
echo "serve.sh: not ready after 600 s" >&2; exit 1

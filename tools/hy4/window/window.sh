#!/bin/bash
# Hy4-Flash GPU window on the Spark: take ~/gpu.lock, stop GLM prod (glm53-tensorfold.service, :8083), run the Hy4
# kernel tests against their CPU references (hy4-gpu-test: hc, the forward layers, Engram, vision; HY4_LOAD=1 adds the
# whole-pack load + spot checks), then serve Hy4 (HY4_SERVE_BIN, --vision) and run the text + image smoke, then restore
# GLM prod and check it (health 200, a chat answer, TC2). Restore runs on success, failure, Ctrl-C, and by a deadline
# (HY4_WINDOW_MIN, default 45) from a detached sleeper; a guard kills the Hy4 processes when MemAvailable drops under
# 2 GiB. NEEDS JC's OK BEFORE RUNNING.
#   usage: HY4_SERVE_BIN=~/hy4gpu/tensorfold-hy4 bash ~/hy4-logs/window.sh   (files: ~/hy4-logs, binaries: ~/hy4gpu)
set -u
D=~/hy4-logs; G=~/hy4gpu; U=glm53-tensorfold.service; P=~/models/Hy4-Flash-SAGE-REXL3-4.84bpw; L=$D/window.log
MIN=${HY4_WINDOW_MIN:-45}
mkdir -p $D
log(){ echo "$(date -u +%T)Z $*" | tee -a $L; }
q(){ echo "- $(date -u +%H:%MZ) hy4: $*" >> ~/GPU-QUEUE.md; }

# --- preflight: nothing changes until all of these hold
[ -e ~/gpu.lock ] && { echo "gpu.lock held: $(cat ~/gpu.lock)"; exit 1; }
[ -x $G/hy4-gpu-test ] && [ -s $G/hc.ptx ] && [ -s $G/hy4_forward.fatbin ] || { echo "missing $G/hy4-gpu-test, $G/hc.ptx or $G/hy4_forward.fatbin"; exit 1; }
[ -z "${HY4_SERVE_BIN:-}" ] || [ -x "$HY4_SERVE_BIN" ] || { echo "HY4_SERVE_BIN=$HY4_SERVE_BIN is not executable"; exit 1; }
[ -s $P/model.safetensors.index.json ] && [ -s $P/engram.safetensors.index.json ] && [ -s $P/config.json ] || { echo "pack incomplete at $P"; exit 1; }
tail -1 ~/models/hy4-verify.log 2>/dev/null | grep -q 'files=25 .* bad=0$' || { echo "~/models/hy4-verify.log does not end in a clean verify (files=25 ... bad=0)"; exit 1; }
code=$(curl -s -o /dev/null -w %{http_code} -m5 http://127.0.0.1:8083/health)
[ "$code" = 200 ] || { echo "GLM prod is not healthy now ($code): not starting a window"; exit 1; }
# Mercury crons: 09:00 Paris daily, Fri 11:00/12:00 Paris (08:00Z/10:00Z/11:00Z in winter); keep the window clear of them
python3 - "$MIN" <<'PY' || { echo "window would overlap a Mercury cron: pick another time"; exit 1; }
import sys, datetime as dt
from zoneinfo import ZoneInfo
z = ZoneInfo("Europe/Paris"); now = dt.datetime.now(z); end = now + dt.timedelta(minutes=int(sys.argv[1]) + 10)
crons = [now.replace(hour=9, minute=0, second=0, microsecond=0)]
if now.isoweekday() == 5: crons += [now.replace(hour=11, minute=0, second=0, microsecond=0), now.replace(hour=12, minute=0, second=0, microsecond=0)]
crons += [c + dt.timedelta(days=1) for c in crons]
sys.exit(1 if any(now - dt.timedelta(minutes=5) <= c <= end for c in crons) else 0)
PY

# --- take
cp ~/tf-prod-glm53/COMMIT $D/prod.COMMIT
echo "hy4 $(date -u +%FT%TZ) pid $$: Hy4-Flash window (feat/hy4-flash: kernel tests + text/image smoke), GLM prod stopped, restored by $(date -u -d +${MIN}min +%H:%MZ)" > ~/gpu.lock
q "took the GPU (~/gpu.lock), GLM prod stop: Hy4-Flash window (feat/hy4-flash: kernel tests vs CPU refs, Hy4 test server on 127.0.0.1:8095 text+image smoke); NO deploy; GLM prod restored by $(date -u -d +${MIN}min +%H:%MZ) at the latest"
log "WINDOW start; GLM prod $(cut -d' ' -f1 $D/prod.COMMIT)"
nohup bash -c "sleep $((MIN * 60)); bash $D/restore.sh deadline" > $D/deadline.out 2>&1 & echo $! > $D/deadline.pid
trap 'bash $D/restore.sh trap' EXIT
systemctl --user stop $U; sleep 3; docker rm -f glm53-prod >/dev/null 2>&1
for i in $(seq 60); do [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 100 ] && break; sleep 2; done
log "GLM prod stopped; $(free -g | awk '/Mem:/{print "MemAvailable " $7 " GiB"}'); $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)"

# --- guard: the Hy4 run dies before the box does
nohup bash -c 'low=999999999; while true; do a=$(awk "/MemAvailable/{print \$2}" /proc/meminfo); [ $a -lt $low ] && { low=$a; echo "$(date -u +%T)Z low-water MemAvailable $((a/1024)) MiB"; }; if [ $a -lt 2097152 ]; then echo "$(date -u +%T)Z MemAvailable < 2 GiB: killing hy4-gpu-test and tensorfold-hy4"; pkill -x hy4-gpu-test; pkill -x tensorfold-hy4; fi; sleep 0.5; done' > $D/guard.out 2>&1 & echo $! > $D/guard.pid

# --- 1. the kernels against their CPU references (hc, forward layers, Engram, vision); HY4_LOAD=1: whole-pack load too
log "kernels: hy4-gpu-test (HY4_FWD_FATBIN, HY4_HC_PTX${HY4_LOAD:+, HY4_LOAD_PACK=$P})"
( cd $G && HY4_FWD_FATBIN=$G/hy4_forward.fatbin HY4_HC_PTX=$G/hc.ptx ${HY4_LOAD:+HY4_LOAD_PACK=$P} HY4_LOAD_RESERVE_MIB=${HY4_RESERVE_MIB:-6144} timeout 1800 nice -n 5 ./hy4-gpu-test ) > $D/kernels.out 2>&1
rc=$?
grep -E "^hy4 |pass|fail|error|differ|expected" $D/kernels.out | tee -a $L
log "kernels rc $rc"

# --- 2. serve Hy4 (text + vision) and smoke it; runs even when a kernel check failed, so the log shows both
if [ -n "${HY4_SERVE_BIN:-}" ]; then
  log "serve: $HY4_SERVE_BIN on 127.0.0.1:8095 (--vision, --context 32768)"
  HY4_RESERVE_MIB=${HY4_SERVE_RESERVE_MIB:-4096} "$HY4_SERVE_BIN" serve $P --name Hy4-Flash --host 127.0.0.1 --port 8095 --context 32768 --parallel 1 --vision > $D/serve.out 2>&1 & echo $! > $D/serve.pid
  up=0
  for i in $(seq 300); do curl -sf -m3 http://127.0.0.1:8095/health >/dev/null && { up=1; break; }; kill -0 $(cat $D/serve.pid) 2>/dev/null || break; sleep 3; done
  grep -E "hy4|error|warn" $D/serve.out | tail -20 | tee -a $L
  if [ $up = 1 ]; then
    log "serve: healthy after ~$((i * 3)) s; $(free -g | awk '/Mem:/{print "MemAvailable " $7 " GiB"}')"
    timeout 1500 python3 $D/smoke.py 8095 $D/smoke.png 2>&1 | tee -a $L
  else
    log "serve: never healthy"
  fi
  kill $(cat $D/serve.pid) 2>/dev/null; for i in $(seq 20); do kill -0 $(cat $D/serve.pid) 2>/dev/null || break; sleep 1; done; kill -9 $(cat $D/serve.pid) 2>/dev/null
else
  log "smoke: skipped (HY4_SERVE_BIN unset)"
fi
log "WINDOW work done; restoring"
# restore runs from the EXIT trap

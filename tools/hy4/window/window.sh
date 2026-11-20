#!/bin/bash
# Hy4-Flash GPU window on the Spark: take ~/gpu.lock, stop GLM prod (glm53-tensorfold.service, :8083), run the Hy4
# kernel tests against their CPU references (hy4-gpu-test: hc, the forward layers incl. ring / bf16 / fp8 caches and
# two sequences a call, Engram, vision; HY4_LOAD=1 adds the whole-pack load + spot checks), then serve Hy4
# (HY4_SERVE_BIN, --vision, 32k, 4 lanes, bf16 caches) for the text + image smoke and the bench (short / 2k / 8k,
# 4 streams at once), then a profiling pass (HY4_PROF_VARIANTS boots: chunk and tensor-core attention on/off, op
# timings from HY4_PROFILE), then the stream capacity: one boot per context in HY4_CAP_CTX (default 256k 512k 1M, caches
# HY4_CAP_DT, default fp8) logging the lanes that fit, MemAvailable, a HY4_FILL_WORDS-word fill (~1.3 tokens a word)
# and min(lanes, 4) streams at once. Then restore GLM prod and check it (health 200, a chat answer, TC2). Restore
# runs on success, failure, Ctrl-C, and by a deadline (HY4_WINDOW_MIN, default 60) from a detached sleeper; a guard
# kills the Hy4 processes when MemAvailable drops under 2 GiB; a capacity boot only starts with 10 minutes left.
# Refuses to start while a Gradle or Kotlin daemon runs (hermes-agent builds hold GBs that GLM needs back).
# NEEDS JC's OK BEFORE RUNNING.
#   usage: HY4_SERVE_BIN=~/hy4gpu/tensorfold-hy4 bash ~/hy4-logs/window.sh   (files: ~/hy4-logs, binaries: ~/hy4gpu)
set -u
D=~/hy4-logs; G=~/hy4gpu; U=glm53-tensorfold.service; P=~/models/Hy4-Flash-SAGE-REXL3-4.84bpw; L=$D/window.log
MIN=${HY4_WINDOW_MIN:-60}
END=$(( $(date +%s) + MIN * 60 - 240 ))  # leave ~4 min of the window for the restore
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
# a Gradle / Kotlin daemon (hermes-agent builds) keeps GBs resident and stops GLM from booting again: refuse, kill nothing
daemons=$(ps -eo pid=,rss=,comm=,args= | awk '$3 == "java" && /GradleDaemon|KotlinCompileDaemon/{printf "pid %s (%d MiB): %s; ", $1, $2/1024, ($0 ~ /GradleDaemon/ ? "GradleDaemon" : "KotlinCompileDaemon")}')
if [ -z "$daemons" ] && docker top hermes-agent >/dev/null 2>&1; then
  daemons=$(docker top hermes-agent -eo pid,rss,comm,args 2>/dev/null | awk 'NR>1 && $3 == "java" && /GradleDaemon|KotlinCompileDaemon/{printf "pid %s (%d MiB, hermes-agent); ", $1, $2/1024}')
fi
[ -z "$daemons" ] || { echo "Gradle/Kotlin daemon running (hermes-agent): $daemons- not starting a window (ask JC before stopping it)"; exit 1; }
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
echo "hy4 $(date -u +%FT%TZ) pid $$: Hy4-Flash window (feat/hy4-flash: kernel tests, text/image smoke, bench, stream capacity 256k-1M), GLM prod stopped, restored by $(date -u -d +${MIN}min +%H:%MZ)" > ~/gpu.lock
q "took the GPU (~/gpu.lock), GLM prod stop: Hy4-Flash window (feat/hy4-flash: kernel tests vs CPU refs, Hy4 test server on 127.0.0.1:8095 text+image smoke + bench + stream capacity at 256k/512k/1M); NO deploy; GLM prod restored by $(date -u -d +${MIN}min +%H:%MZ) at the latest"
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
( cd $G && env HY4_FWD_FATBIN=$G/hy4_forward.fatbin HY4_HC_PTX=$G/hc.ptx ${HY4_LOAD:+HY4_LOAD_PACK=$P} HY4_LOAD_RESERVE_MIB=${HY4_RESERVE_MIB:-6144} timeout ${HY4_KT_TIMEOUT:-600} nice -n 5 ./hy4-gpu-test ) > $D/kernels.out 2>&1
rc=$?
grep -E "^hy4 |pass|fail|error|differ|expected" $D/kernels.out | tee -a $L
log "kernels rc $rc"

# serve_up CTX EXTRA_ARGS...: Hy4 on 127.0.0.1:8095 with --vision at CTX tokens; 0 once healthy
serve_up(){
  local ctx=$1; shift
  log "serve: $HY4_SERVE_BIN --context $ctx $* (cache ${HY4_CACHE_DT:-bf16})"
  env HY4_RESERVE_MIB=${HY4_SERVE_RESERVE_MIB:-4096} "$HY4_SERVE_BIN" serve $P --name Hy4-Flash --host 127.0.0.1 --port 8095 --context $ctx --vision "$@" > $D/serve.out 2>&1 & echo $! > $D/serve.pid
  local up=0 i
  for i in $(seq 300); do curl -sf -m3 http://127.0.0.1:8095/health >/dev/null && { up=1; break; }; kill -0 $(cat $D/serve.pid) 2>/dev/null || break; sleep 3; done
  grep -E "hy4|error|warn" $D/serve.out | grep -v "hy4 load:" | tail -20 | tee -a $L
  cp $D/serve.out $D/serve-$ctx${HY4_CACHE_DT:+-$HY4_CACHE_DT}.out
  if [ $up = 1 ]; then log "serve: healthy after ~$((i * 3)) s; $(free -m | awk '/Mem:/{print "MemAvailable " $7 " MiB"}')"; return 0; fi
  log "serve: never healthy at $ctx"; return 1
}
serve_down(){
  kill $(cat $D/serve.pid) 2>/dev/null; for i in $(seq 20); do kill -0 $(cat $D/serve.pid) 2>/dev/null || break; sleep 1; done; kill -9 $(cat $D/serve.pid) 2>/dev/null
  for i in $(seq 30); do pgrep -x tensorfold-hy4 >/dev/null || break; sleep 1; done
}

# --- 2. serve Hy4 (text + vision, 32k, 4 lanes) and smoke + bench it; runs even when a kernel check failed
if [ -n "${HY4_SERVE_BIN:-}" ]; then
  if HY4_MOE_CHECK=${HY4_MOE_CHECK:-4} serve_up 32768 --parallel 4; then
    grep -E "moe check" $D/serve.out | tail -4 | tee -a $L
    timeout 1500 python3 $D/smoke.py 8095 $D/smoke.png 2>&1 | tee -a $L
    grep -E "moe check" $D/serve.out | tail -4 | tee -a $L
    [ -s $D/bench.py ] && timeout $(( END - $(date +%s) - ${HY4_CAP_RESERVE_S:-900} > 180 ? END - $(date +%s) - ${HY4_CAP_RESERVE_S:-900} : 180 )) python3 $D/bench.py 8095 --parallel 4 2>&1 | tee -a $L
    log "serve: $(free -m | awk '/Mem:/{print "MemAvailable " $7 " MiB"}') after the smoke + bench"
    # HY4_HOLD=1: keep the server up for more tests until ~/hy4-logs/release exists (the deadline still restores)
    if [ -n "${HY4_HOLD:-}" ]; then rm -f $D/release; log "hold: touch $D/release to end"; while [ ! -e $D/release ] && kill -0 $(cat $D/serve.pid) 2>/dev/null; do sleep 5; done; fi
  fi
  serve_down

  # --- 2b. profiling pass: one 32k boot per variant in HY4_PROF_VARIANTS (CHUNK:ATTN_TC_ROWS, default the new path at
  # chunks 256 and 512 and the scalar attention at 256), HY4_PROFILE=4 (op groups timed on the stream: a summary every 4
  # steps), a ~6k-token prompt then 4 streams; the "hy4 profile" lines go to the window log. Skipped with < 15 min left
  # before the capacity reserve. HY4_PROF_VARIANTS= (empty) skips it; HY4_CAP_CTX=" " (a space) skips the capacity.
  for v in ${HY4_PROF_VARIANTS-256:16 512:16 256:0}; do
    left=$(( END - $(date +%s) - ${HY4_CAP_RESERVE_S:-900} ))
    [ $left -ge 900 ] || { log "profile $v: skipped, $((left / 60)) min left before the capacity reserve"; continue; }
    if HY4_PROFILE=${HY4_PROFILE_EVERY:-4} HY4_CHUNK=${v%%:*} HY4_ATTN_TC_ROWS=${v##*:} serve_up 32768 --parallel 4; then
      timeout $(( left > 900 ? 600 : left - 300 )) python3 $D/bench.py 8095 --skip-base --fill ${HY4_PROF_FILL:-4600} --parallel 4 2>&1 | sed "s/^/[chunk ${v%%:*} tc ${v##*:}] /" | tee -a $L
      grep -E "hy4 profile" $D/serve.out | sed "s/^/[chunk ${v%%:*} tc ${v##*:}] /" | tail -${HY4_PROF_LINES:-40} | tee -a $L
      cp $D/serve.out $D/serve-prof-${v%%:*}-${v##*:}.out
    else
      log "profile $v: boot failed; $(grep -E 'error|budget|OutOfDeviceMemory' $D/serve.out | tail -2 | tr '\n' ' ')"
    fi
    serve_down
  done

  # --- 3. stream capacity: a boot per context, lanes not fixed (the engine takes what fits, at most 8)
  for ctx in ${HY4_CAP_CTX:-262144 524288 1048576}; do
    left=$(( END - $(date +%s) ))
    [ $left -ge 600 ] || { log "capacity $ctx: skipped, $((left / 60)) min left in the window"; continue; }
    export HY4_CACHE_DT=${HY4_CAP_DT:-fp8}
    if serve_up $ctx; then
      lanes=$(grep -oE "hy4: [0-9]+ lanes of" $D/serve.out | head -1 | awk '{print $2}')  # not the 4 of "hy4"
      log "CAPACITY $ctx tokens ($HY4_CACHE_DT caches): ${lanes:-?} lanes; $(grep -E 'hy4: [0-9]+ lanes|ready, work' $D/serve.out | tr '\n' ' ')"
      grep -E "hy4 capacity:" $D/serve.out | tee -a $L
      k=${lanes:-1}; [ "$k" -gt 4 ] && k=4
      left=$(( END - $(date +%s) ))
      fill=${HY4_FILL_WORDS:-24000}; [ $left -ge 480 ] || fill=4000
      timeout $(( left > 120 ? left - 60 : 60 )) python3 $D/bench.py 8095 --skip-base --fill $fill --parallel $k 2>&1 | tee -a $L
      log "CAPACITY $ctx: $(free -m | awk '/Mem:/{print "MemAvailable " $7 " MiB"}') after the fill and $k streams; $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
    else
      grep -E "budget|fits|OutOfDeviceMemory" $D/serve.out | tail -3 | tee -a $L
    fi
    serve_down
    unset HY4_CACHE_DT
  done
else
  log "smoke: skipped (HY4_SERVE_BIN unset)"
fi
log "WINDOW work done; restoring"
# restore runs from the EXIT trap

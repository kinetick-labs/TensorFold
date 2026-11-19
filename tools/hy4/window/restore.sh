#!/bin/bash
# restore.sh WHY: stop anything Hy4, bring GLM prod (glm53-tensorfold.service, :8083) back, check health + chat + TC2,
# release ~/gpu.lock. Safe to run twice (flock; only acts while the lock is ours). When GLM fails to boot for memory
# ("budget fits no stream", a CUDA budget or out-of-memory line in its journal) it says so plainly in window.log,
# GPU-QUEUE.md and ~/hy4-logs/RESTORE-FAILED, naming the biggest processes (and any Gradle/Kotlin daemon); it kills
# nothing of anyone else's.
D=~/hy4-logs; U=glm53-tensorfold.service; L=$D/window.log
exec 9>$D/restore.flock; flock -n 9 || { echo "restore already running"; exit 0; }
[ -e ~/gpu.lock ] && grep -q '^hy4 ' ~/gpu.lock || { echo "lock not ours / already released"; exit 0; }
log(){ echo "$(date -u +%T)Z $*" | tee -a $L; }
q(){ echo "- $(date -u +%H:%MZ) hy4: $*" >> ~/GPU-QUEUE.md; }
log "RESTORE ($1)"
pkill -x hy4-gpu-test 2>/dev/null; pkill -x tensorfold-hy4 2>/dev/null; [ -s $D/serve.pid ] && kill -9 $(cat $D/serve.pid) 2>/dev/null
kill $(cat $D/guard.pid 2>/dev/null) 2>/dev/null
[ "$1" != deadline ] && { p=$(cat $D/deadline.pid 2>/dev/null); [ -n "$p" ] && pkill -P $p 2>/dev/null; kill $p 2>/dev/null; }
for i in $(seq 30); do pgrep -x hy4-gpu-test >/dev/null || pgrep -x tensorfold-hy4 >/dev/null || break; sleep 1; done
rm -f $D/RESTORE-FAILED
memfail(){ journalctl --user -u $U --since "@$t0" --no-pager 2>/dev/null | grep -iE "budget fits no stream|memory budget leaves|OutOfDeviceMemory|out of memory|CUDA_ERROR_OUT_OF_MEMORY|cannot allocate" | tail -3; }
t0=$(date +%s); log "before GLM start: $(free -m | awk '/Mem:/{print "MemAvailable " $7 " MiB"}')"; systemctl --user start $U
mem=""
for i in $(seq 400); do
  curl -sf -m3 http://127.0.0.1:8083/health >/dev/null && break
  mem=$(memfail); [ -n "$mem" ] && break
  systemctl --user is-failed -q $U 2>/dev/null && break
  sleep 3
done
code=$(curl -s -o /dev/null -w %{http_code} -m5 http://127.0.0.1:8083/health)
log "GLM prod health $code after $(( $(date +%s) - t0 )) s ($(cut -d' ' -f1 ~/tf-prod-glm53/COMMIT))"
if [ "$code" != 200 ]; then
  [ -n "$mem" ] || mem=$(memfail)
  if [ -n "$mem" ]; then
    hogs=$(ps -eo pid=,rss=,comm= --sort=-rss | head -6 | awk '{printf "%s pid %s %d MiB; ", $3, $1, $2/1024}')
    daemons=$(ps -eo pid=,rss=,comm=,args= | awk '$3 == "java" && /GradleDaemon|KotlinCompileDaemon/{printf "pid %s %d MiB %s; ", $1, $2/1024, ($0 ~ /GradleDaemon/ ? "GradleDaemon" : "KotlinCompileDaemon")}')
    avail=$(free -m | awk '/Mem:/{print $7}')
    msg="GLM PROD DID NOT BOOT FOR MEMORY ($1): MemAvailable ${avail} MiB; GLM said: $(echo "$mem" | tail -1 | sed 's/^.*\]: //' | cut -c1-240); biggest processes: $hogs${daemons:+ Gradle/Kotlin daemons: $daemons}- nothing was killed: free that memory (ask its owner), then 'systemctl --user restart $U'"
    log "$msg"; q "$msg"; echo "$msg" > $D/RESTORE-FAILED
    echo "$msg" >&2
  fi
fi
chat=$(python3 - <<'PY' 2>&1
import json, urllib.request
m = json.load(urllib.request.urlopen("http://127.0.0.1:8083/v1/models", timeout=10))["data"][0]["id"]
b = {"model": m, "messages": [{"role": "user", "content": "Reply with exactly: OK"}], "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8083/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=300))
print((r["choices"][0]["message"].get("content") or "").strip()[:40])
PY
)
chat=$(echo "$chat" | tail -1 | cut -c1-120)
log "GLM chat: '$chat'"
python3 ~/tf-mia-logs/df2g/tc2.py 8083 > $D/restore-tc2.out 2>&1; n=$(grep -c OK $D/restore-tc2.out); log "TC2 $n/7"
[ "$code" = 200 ] && [ "$n" -ge 6 ] || { log "WARNING: GLM prod not fully back (health $code, TC2 $n/7): check it now"; [ -e $D/RESTORE-FAILED ] || echo "GLM prod not fully back ($1): health $code, TC2 $n/7" > $D/RESTORE-FAILED; }
q "GLM prod $(cut -d' ' -f1 ~/tf-prod-glm53/COMMIT) $([ "$code" = 200 ] && echo restored || echo "NOT back") $(date -u +%TZ) ($1): health $code, chat '$chat', TC2 $n/7; released ~/gpu.lock"
rm -f ~/gpu.lock; log "lock released"

#!/bin/bash
# kernel-beside-prod.sh: the Hy4 kernel tests (hy4-gpu-test, HY4_HC_PTX only) beside GLM prod: a few MiB on the GPU, under ~/gpu.lock, only with MemAvailable >= 4.5 GiB; prod untouched.
q(){ echo "- $(date -u +%H:%MZ) hy4: $*" >> ~/GPU-QUEUE.md; }
[ -e ~/gpu.lock ] && { echo "gpu.lock held: $(cat ~/gpu.lock)"; exit 1; }
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
[ "$avail" -ge 4608 ] || { echo "MemAvailable ${avail} MiB < 4608: not running beside prod"; exit 1; }
echo "hy4 $(date -u +%FT%TZ) pid $$: Hy4 hc kernel unit test beside prod (few MiB, seconds); prod NOT stopped" > ~/gpu.lock
q "took the GPU (~/gpu.lock) beside prod: Hy4 hc kernel unit test (few MiB, seconds); prod :8083 NOT stopped"
cd ~/hy4gpu
HY4_HC_PTX=$PWD/hc.ptx systemd-run --user --scope -q -E HY4_HC_PTX=$PWD/hc.ptx -p MemoryMax=1500M nice -n 10 ./hy4-gpu-test 2>&1 | tail -5
rc=${PIPESTATUS[0]}
code=$(curl -s -o /dev/null -w %{http_code} -m5 http://127.0.0.1:8083/health)
rm -f ~/gpu.lock
q "Hy4 hc kernel test done (rc $rc), prod health $code; released ~/gpu.lock"
echo "rc=$rc prod=$code MemAvailable before ${avail} MiB, after $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"

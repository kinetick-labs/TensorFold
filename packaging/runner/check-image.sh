#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then echo 'usage: check-image.sh TAG' >&2; exit 2; fi
tag=$1
run() { docker run --rm --runtime=runc --network none -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= "$@"; }
run "$tag" /bin/true
run -e TF_RUNNER=1 "$tag" /bin/true
for dependency in TF_RUNNER_CLANG TF_RUNNER_LLD; do
 run -e "$dependency=/nonexistent-runner-tool" -e TF_RUNNER=0 "$tag" /bin/true
 rc=0
 run -e "$dependency=/nonexistent-runner-tool" -e TF_RUNNER=1 "$tag" /bin/true || rc=$?
 [[ $rc != 0 ]] || { echo "FAIL: missing $dependency did not fail closed" >&2; exit 1; }
done
rc=0
run -e TF_RUNNER_CLANG=/bin/true -e TF_RUNNER=1 "$tag" /bin/true || rc=$?
[[ $rc != 0 ]] || { echo 'FAIL: compiler success without an ELF was accepted' >&2; exit 1; }
run "$tag" /bin/sh -c 'test "$1" = "hello world"' sh 'hello world'
circuit=/opt/tensorfold/share/runner/calc.tfpc
probed=no
if run "$tag" test -f "$circuit"; then
 run --entrypoint /opt/tensorfold/bin/tf-runner-frontend-check "$tag" --require --circuit "$circuit" ||
  { echo "FAIL: the image's circuit route (compiled probe cells on $circuit) is wrong" >&2; exit 1; }
 probed=yes
fi
echo "RUNNER_IMAGE_CHECK: compile/ELF validation, optional fallback, required failure, argv preservation, circuit route ($probed) PASS"

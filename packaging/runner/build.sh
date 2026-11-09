#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then echo 'usage: build.sh BASE_IMAGE TAG BOOT_CHECK_BINARY' >&2; exit 2; fi
base=$1 tag=$2 binary=$3
here=$(cd -- "$(dirname -- "$0")" && pwd)
context=$(mktemp -d -t tf-runner-image.XXXXXXXX)
cleanup() { find "$context" -maxdepth 1 -type f -delete; rmdir "$context"; }
trap cleanup EXIT
cp "$here/Dockerfile" "$context/Dockerfile"
cp "$binary" "$context/tf-runner-frontend-check"
chmod 755 "$context/tf-runner-frontend-check"
docker image inspect "$base" > "$context/base.json"
base_id=$(docker image inspect -f '{{.Id}}' "$base")
python3 - "$context" <<'PY'
import json, pathlib, shlex, sys
root=pathlib.Path(sys.argv[1])
config=json.loads((root/'base.json').read_text())[0]['Config']
entry=config.get('Entrypoint') or []
if '/opt/tensorfold/bin/runner-entrypoint' in entry:
    raise SystemExit('base already contains runner-entrypoint; use the original serving base')
script='#!/bin/sh\nset -eu\n/opt/tensorfold/bin/tf-runner-frontend-check\nexec '
script+=' '.join(shlex.quote(arg) for arg in entry)
script+=' "$@"\n'
(root/'runner-entrypoint').write_text(script)
(root/'runner-entrypoint').chmod(0o755)
with (root/'Dockerfile').open('a') as f:
    f.write('CMD '+json.dumps(config.get('Cmd') or [])+'\n')
PY
DOCKER_BUILDKIT=0 nice -n 19 ionice -c3 docker build \
    --build-arg "BASE_IMAGE=$base_id" \
    --build-arg "FRONTEND_TREE=${TF_RUNNER_FRONTEND_TREE:-unknown}" \
    --build-arg "RUNNER_BUILD_ID=${TF_RUNNER_BUILD_ID:-runner-frontend}" \
    -t "$tag" "$context" >&2
docker image inspect "$tag" > "$context/derived.json"
python3 - "$context" "$tag" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
base=json.loads((root/'base.json').read_text())[0]
image=json.loads((root/'derived.json').read_text())[0]
assert (image['Config'].get('Cmd') or []) == (base['Config'].get('Cmd') or [])
assert image['RootFS']['Layers'][:len(base['RootFS']['Layers'])] == base['RootFS']['Layers']
print(json.dumps({'size_metric':'docker image inspect .Size','tag':sys.argv[2],'image_id':image['Id'],'base_id':base['Id'],
 'base_bytes':base['Size'],'image_bytes':image['Size'],'delta_bytes':image['Size']-base['Size'],
 'frontend_tree':image['Config']['Labels']['org.tensorfold.runner.frontend.tree']},sort_keys=True))
PY

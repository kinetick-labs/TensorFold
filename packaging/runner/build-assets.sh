#!/usr/bin/env bash
# Layer the runner's circuit and graph bank (assets.Dockerfile) over an image build.sh built. Without ASSETS_DIR the
# assets are downloaded from assets.release; either way assets.py checks them against assets.sha256 before the build.
set -euo pipefail
usage='usage: build-assets.sh RUNNER_IMAGE TAG [ASSETS_DIR]'
if [[ $# -lt 2 || $# -gt 3 ]]; then echo "$usage" >&2; exit 2; fi
base=$1 tag=$2 assets=${3:-}
[[ -z "$assets" || -d "$assets" ]] || { echo 'ASSETS_DIR must be a directory' >&2; exit 2; }
here=$(cd -- "$(dirname -- "$0")" && pwd)
context=$(mktemp -d -t tf-runner-assets.XXXXXXXX)
trap 'python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1])" "$context"' EXIT
cp "$here/assets.Dockerfile" "$context/Dockerfile"
if [[ -n "$assets" ]]; then
    cp -R "$assets" "$context/runner"
    python3 "$here/assets.py" check "$context/runner"
else
    python3 "$here/assets.py" fetch "$context/runner"
fi
DOCKER_BUILDKIT=0 nice -n 19 ionice -c3 docker build --build-arg "BASE_IMAGE=$base" -t "$tag" "$context" >&2
docker image inspect -f '{{.Id}}' "$tag"

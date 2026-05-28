#!/usr/bin/env bash
# Build the prebaked agent runtime image. Run once; reuse across tasks via
#   python -m agent '<goal>' --docker --docker-persistent --docker-image crucible-runtime
set -euo pipefail

IMAGE="${1:-crucible-runtime}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building ${IMAGE} from ${DIR}/Dockerfile ..."
docker build -t "${IMAGE}" "${DIR}"
echo "Done. Use it with:  --docker-image ${IMAGE}"

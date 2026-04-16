#!/usr/bin/env bash

# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="gfloat-linux-386:py310-uv"
PLATFORM="linux/386"
DOCKERFILE="etc/linux-386.Dockerfile"
MODE="${1:-all}"

usage() {
    echo "Usage: bash etc/test-linux-386.sh [load|build|run]"
    echo "  load  Build image only if missing"
    echo "  build Force rebuild image"
    echo "  run   Run tests (image must already exist)"
    echo "  (no arg) Build image if missing, then run tests"
}

build_image() {
    docker buildx build \
    --progress=plain \
    --platform "${PLATFORM}" \
    --load \
    -t "${IMAGE}" \
    -f "${REPO_DIR}/${DOCKERFILE}" \
    "${REPO_DIR}"
}

run_tests() {
    docker run --rm \
    --platform "${PLATFORM}" \
    -v "${REPO_DIR}:/work" \
    -w /work \
    "${IMAGE}" \
    bash -lc '
        set -euo pipefail
        export PYTHONPATH="/work/src${PYTHONPATH:+:${PYTHONPATH}}"
        python -m pytest -vv test
    '
}

if [[ "${MODE}" != "all" && "${MODE}" != "load" && "${MODE}" != "build" && "${MODE}" != "run" ]]; then
    usage
    exit 2
fi

echo "Running full unit tests in Docker (${PLATFORM})"

if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is not running; start Docker and rerun this script." >&2
    exit 1
fi

if [[ "${MODE}" == "build" ]]; then
    echo "Force-building test image ${IMAGE} for ${PLATFORM}"
    build_image
fi

if [[ "${MODE}" == "load" || "${MODE}" == "all" ]]; then
    if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
        echo "Building test image ${IMAGE} for ${PLATFORM} (cached after first build)"
        build_image
    fi
fi

if [[ "${MODE}" == "run" || "${MODE}" == "all" ]]; then
    if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
        echo "Image ${IMAGE} not found. Run 'bash etc/test-linux-386.sh load' first." >&2
        exit 1
    fi

    run_tests
fi

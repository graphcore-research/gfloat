#!/usr/bin/env bash

# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${GFLOAT_TEST_IMAGE:?GFLOAT_TEST_IMAGE is required}"
PLATFORM="${GFLOAT_TEST_PLATFORM:?GFLOAT_TEST_PLATFORM is required}"
DOCKERFILE="${GFLOAT_TEST_DOCKERFILE:?GFLOAT_TEST_DOCKERFILE is required}"
SCRIPT_NAME="${GFLOAT_TEST_SCRIPT_NAME:-$(basename "$0")}"

MODE="all"
if [[ "${1:-}" == "all" || "${1:-}" == "load" || "${1:-}" == "build" || "${1:-}" == "run" ]]; then
    MODE="$1"
    shift
fi

usage() {
    echo "Usage: bash etc/${SCRIPT_NAME} [all|load|build|run] [pytest args...]"
    echo "  load  Build image only if missing"
    echo "  build Force rebuild image"
    echo "  run   Run tests (image must already exist)"
    echo "  (no arg) Build image if missing, then run tests"
    echo ""
    echo "Examples:"
    echo "  bash etc/${SCRIPT_NAME} run test/test_encode.py::test_encode[binary32]"
    echo "  bash etc/${SCRIPT_NAME} run -k stochastic -q"
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
        python -m pytest "$@"
    ' _ "$@"
}

if [[ "${MODE}" != "all" && "${MODE}" != "load" && "${MODE}" != "build" && "${MODE}" != "run" ]]; then
    usage
    exit 2
fi

echo "Running unit tests in Docker (${PLATFORM})"

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
        echo "Image ${IMAGE} not found. Run 'bash etc/${SCRIPT_NAME} load' first." >&2
        exit 1
    fi

    run_tests "$@"
fi

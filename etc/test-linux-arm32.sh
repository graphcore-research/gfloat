#!/usr/bin/env bash

# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GFLOAT_TEST_IMAGE="gfloat-linux-arm32:py310-uv"
export GFLOAT_TEST_PLATFORM="linux/arm/v7"
export GFLOAT_TEST_DOCKERFILE="etc/linux-container.Dockerfile"
export GFLOAT_TEST_SCRIPT_NAME="test-linux-arm32.sh"

exec bash "${SCRIPT_DIR}/test-linux-container.sh" "$@"

# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip uv \
    && uv pip install --system \
        "numpy<2" \
        pytest \
        nbval \
        ml_dtypes \
        array-api-compat \
        array-api-strict \
        more_itertools

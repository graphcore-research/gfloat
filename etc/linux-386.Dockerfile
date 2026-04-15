# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip uv

RUN NINJAFLAGS='-v' python -m pip install -v --no-cache-dir \
    "numpy<2"

RUN NINJAFLAGS='-v' UV_NO_PROGRESS=1 uv pip install --system \
    psutil \
    ml_dtypes

RUN UV_NO_PROGRESS=1 uv pip install --system \
    pytest \
    nbval \
    array-api-compat \
    array-api-strict \
    more_itertools

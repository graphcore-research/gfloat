# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN NINJAFLAGS='-v' python -m pip install -v --no-build-isolation --no-cache-dir \
    "numpy<2" \
    psutil \
    ml_dtypes \
    pytest \
    nbval \
    array-api-compat \
    array-api-strict \
    more_itertools

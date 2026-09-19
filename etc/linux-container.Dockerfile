# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U --no-cache-dir pip

RUN NINJAFLAGS='-v' python -m pip install -v --no-cache-dir \
    "numpy<2"

# Build against the installed NumPy 1.x on 32-bit platforms. ml_dtypes 0.6
# requires NumPy 2 and uses a different build backend.
RUN python -m pip install --no-cache-dir "setuptools~=80.9.0" \
    && NINJAFLAGS='-v' python -m pip install -v --no-build-isolation --no-deps --no-cache-dir \
    "ml_dtypes==0.5.4"

RUN NINJAFLAGS='-v' python -m pip install -v --no-cache-dir \
    psutil \
    pytest \
    nbval \
    array-api-compat \
    array-api-strict \
    more_itertools

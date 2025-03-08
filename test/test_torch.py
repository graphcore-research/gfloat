# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import numpy as np

import torch

import gfloat
from gfloat.formats import *


def test_torch() -> None:
    """
    Test that Torch tensors work
    """
    a = torch.randn(1024)

    a8 = a.to(dtype=torch.float8_e5m2).to(dtype=torch.float32)

    fi = format_info_ocp_e5m2
    t8 = gfloat.round_ndarray(fi, a)  # type: ignore [arg-type]

    torch.testing.assert_close(a8, t8, atol=0.0, rtol=0.0)

    tc = torch.compile(lambda x: gfloat.round_ndarray(fi, x))
    t8i = tc(a)

    torch.testing.assert_close(a8, t8i, atol=0.0, rtol=0.0)

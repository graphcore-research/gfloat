# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import pytest

import numpy as np

import torch

from mx.mx_ops import quantize_mx_op
from mx.formats import ElemFormat


from gfloat import BlockFormatInfo, RoundMode, quantize_block, compute_scale_amax
from gfloat.formats import *


@pytest.mark.parametrize(
    ("mx_round,gf_round"),
    [("even", RoundMode.TiesToEven), ("nearest", RoundMode.TiesToAway)],
)
@pytest.mark.parametrize(
    ("mx_etype,gf_etype"),
    [
        (ElemFormat.int8, format_info_ocp_int8),
        (ElemFormat.fp6_e3m2, format_info_ocp_e3m2),
        (ElemFormat.fp4_e2m1, format_info_ocp_e2m1),
    ],
)
def test_mx(
    mx_round: str,
    gf_round: RoundMode,
    mx_etype: ElemFormat,
    gf_etype: FormatInfo,
) -> None:
    # Input tensor
    A = np.arange(32) / 2 - 5

    # MX: Declare block format
    mx_specs = dict(
        block_size=32,
        scale_bits=8,
        shared_exp_method="max",
        mx_flush_fp32_subnorms=False,
        custom_cuda=False,
    )

    # MX: Quantize
    mx_dq = quantize_mx_op(torch.tensor(A), mx_specs, mx_etype, axes=0, round=mx_round)

    # GFloat: Declare block format
    fi = BlockFormatInfo("test", gf_etype, 32, format_info_ocp_e8m0)

    # GFloat: Quantize
    gf_dq = quantize_block(fi, A, compute_scale_amax, gf_round)

    # Compare
    np.testing.assert_allclose(gf_dq, mx_dq)

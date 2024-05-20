# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import pytest

import numpy as np
from numpy.typing import NDArray

import torch

from mx.mx_ops import quantize_mx_op
from mx.formats import ElemFormat


from gfloat import (
    BlockFormatInfo,
    RoundMode,
    quantize_block,
    compute_scale_amax,
    encode_block,
)
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
@pytest.mark.parametrize(
    "A",
    [
        np.arange(32) / 2 - 5,
        np.zeros(32),
    ],
    ids=[
        "tennish",
        "zeros",
    ],
)
def test_mx(
    mx_etype: ElemFormat,
    gf_etype: FormatInfo,
    mx_round: str,
    gf_round: RoundMode,
    A: NDArray[np.float64],
) -> None:
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


def test_mx_exceptions() -> None:
    fi = BlockFormatInfo("test", format_info_ocp_e2m1, 32, format_info_ocp_e8m0)

    A = np.ones(32) * 2.0**-139

    s = compute_scale_amax(fi.etype.emax, A)
    assert s == 2.0**-127

    with pytest.raises(ValueError, match="out of range"):
        list(encode_block(fi, fi.stype.max * 2, A))

    assert not fi.stype.is_signed
    scale = fi.stype.min / 2
    assert scale != 0
    with pytest.raises(ValueError, match="out of range"):
        list(encode_block(fi, scale, A))

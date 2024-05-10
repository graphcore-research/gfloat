# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import pytest

import numpy as np

import torch

from mx.mx_ops import quantize_mx_op
from mx.formats import ElemFormat


from gfloat import (
    BlockFormatInfo,
    encode_block,
    decode_block,
    encode_float,
    decode_float,
    round_float,
    RoundMode,
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
    ids=str,
)
def test_mx(
    mx_round: str,
    gf_round: RoundMode,
    mx_etype: ElemFormat,
    gf_etype: FormatInfo,
) -> None:
    ## Input tensor
    A = np.arange(32) / 2 - 5

    ## Compute MX quantization
    # Declare block format
    mx_specs = dict(
        block_size=32,
        scale_bits=8,
        shared_exp_method="max",
        mx_flush_fp32_subnorms=False,
        custom_cuda=False,
    )

    # Compute scale, encode, decode
    mx_dq = quantize_mx_op(torch.tensor(A), mx_specs, mx_etype, axes=0, round=mx_round)

    ## Compute GFloat quantization
    # Declare block format
    fi = BlockFormatInfo("test", gf_etype, 32, format_info_ocp_e8m0)

    # Compute scale - this is not considered GFloat's job, but could easily be added
    amax = np.max(np.abs(A))
    q_log2scale = np.floor(np.log2(amax)) - fi.etype.emax
    q_scale = 2**q_log2scale

    # Apply scale to encode and decode
    enc = encode_block(fi, q_scale, A, gf_round)
    gf_dq = list(decode_block(fi, enc))

    ## Compare
    np.testing.assert_allclose(gf_dq, mx_dq)

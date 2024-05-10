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
        (ElemFormat.fp6_e3m2, format_info_ocp_e3m2),
        (ElemFormat.fp4_e2m1, format_info_ocp_e2m1),
    ],
    ids=str,
)
def test_mx(mx_round, gf_round, mx_etype, gf_etype):
    A = torch.arange(32) / 2 - 5

    mx_specs = dict(
        block_size=32,
        scale_bits=8,
        shared_exp_method="max",
        mx_flush_fp32_subnorms=False,
        custom_cuda=False,
    )

    mx_dq = quantize_mx_op(A, mx_specs, mx_etype, axes=0, round=mx_round)

    fi = BlockFormatInfo("test", gf_etype, 32, format_info_ocp_e8m0)

    amax = A.abs().max()
    q_log2scale = torch.floor(torch.log2(amax)).item() - fi.etype.emax
    q_scale = 2**q_log2scale

    print(f"{q_scale=}")

    enc = list(encode_block(fi, q_scale, (a.item() for a in A), gf_round))
    print(f"{enc=}")
    print("decoded_scale=", decode_float(fi.stype, enc[0]).fval)
    print("decoded_vals=", list(decode_float(fi.etype, e).fval for e in enc[1:]))
    print(
        "all_vals=",
        *(
            str(decode_float(fi.etype, i).fval) + ("" if i & 1 else "e")
            for i in range(fi.etype.code_of_max + 1)
        ),
    )
    gf_dq = list(decode_block(fi, enc))
    print("input=", *(str(v.item()) for v in A))
    print("mx_dq=", *(str(v.item()) for v in mx_dq))
    print("gf_dq=", *(str(v) for v in gf_dq))

    np.testing.assert_allclose(gf_dq, mx_dq)

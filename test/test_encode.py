# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import ml_dtypes
import numpy as np
import pytest

from gfloat import decode_float, encode_float
from gfloat.formats import *


@pytest.mark.parametrize("fi", all_formats)
def test_encode(fi: FormatInfo) -> None:
    dec = lambda v: decode_float(fi, v).fval

    if fi.bits <= 8:
        step = 1
    elif fi.bits <= 16:
        step = 13
    elif fi.bits <= 32:
        step = 73013
    elif fi.bits <= 64:
        step = (73013 << 32) + 39

    for i in range(0, 2**fi.bits, step):
        fv = decode_float(fi, i)
        code = encode_float(fi, fv.fval)
        assert (i == code) or np.isnan(fv.fval)
        fv2 = decode_float(fi, code)
        np.testing.assert_equal(fv2.fval, fv.fval)


@pytest.mark.parametrize("fi", all_formats)
def test_encode_edges(fi: FormatInfo) -> None:
    assert encode_float(fi, fi.max) == fi.code_of_max

    assert encode_float(fi, fi.max * 1.25) == (
        fi.code_of_posinf
        if fi.has_infs
        else fi.code_of_nan if fi.num_nans > 0 else fi.code_of_max
    )

    if fi.is_signed:
        assert encode_float(fi, fi.min * 1.25) == (
            fi.code_of_neginf
            if fi.has_infs
            else fi.code_of_nan if fi.num_nans > 0 else fi.code_of_min
        )

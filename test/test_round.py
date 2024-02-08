# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import pytest
import ml_dtypes
import numpy as np

from gfloat import decode_float, round_float
from gfloat.formats import *


def test_round_p3109():
    fi = format_info_p3109(4)
    assert round_float(fi, 0.0029296875) == 0.0029296875
    assert round_float(fi, 224.0) == 224.0
    assert round_float(fi, 240.0) == np.inf
    assert round_float(fi, 232.0) == 224.0
    assert round_float(fi, 232.1) == np.inf


def mlround(v, dty):
    return np.array([v]).astype(dty).astype(float).item()


def linterp(a, b, t):
    return a * (1 - t) + b * t


p3109_formats = [format_info_p3109(p) for p in range(2, 7)]

some_positive_codepoints = (
    0x00,
    0x01,
    0x02,
    0x03,
    0x07,
    0x0F,
    0x17,
    0x21,
    0x33,
    0x40,
    0x53,
    0x65,
    0x70,
)


@pytest.mark.parametrize("i", some_positive_codepoints)
@pytest.mark.parametrize(
    "fi",
    [
        format_info_ocp_e5m2,
        format_info_ocp_e4m3,
        *p3109_formats,
    ],
    ids=str,
)
def test_round(fi, i):
    """
    Test rounding from values between exact binary8 values
    For integer code point i, let
      v0 = the float value at i
      v1 = the float value at i+1, i.e. nextUp(v0)
      dv = v1 - v0
    Then check that:
        round(v0) == v0
        round(v0 + 0.3*dv) == v0
        round(v0 + 0.6*dv) == v1
    """
    v0 = decode_float(fi, i + 0).fval
    v1 = decode_float(fi, i + 1).fval
    if np.isfinite([v0, v1]).all():
        dv = v1 - v0
        np.testing.assert_equal(round_float(fi, v0), v0)
        np.testing.assert_equal(round_float(fi, v0 + 0.3 * dv), v0)
        np.testing.assert_equal(round_float(fi, v0 + 0.49 * dv), v0)
        np.testing.assert_equal(round_float(fi, v0 + 0.51 * dv), v1)
        np.testing.assert_equal(round_float(fi, v0 + 0.99 * dv), v1)


test_formats = [
    (format_info_ocp_e5m2, ml_dtypes.float8_e5m2),
    (format_info_ocp_e4m3, ml_dtypes.float8_e4m3fn),
]


@pytest.mark.parametrize("fi,mldtype", test_formats)
def test_ml_dtype_compatible(fi, mldtype):
    """
    Test that rounding is compatible with ml_dtypes
    """
    for i in range(255):
        v0 = decode_float(fi, i + 0).fval
        v1 = decode_float(fi, i + 1).fval

        for alpha in np.arange(0, 1, 0.3):
            v = linterp(v0, v1, alpha)

            print(i)
            val = round_float(fi, v)

            mlval = mlround(v, mldtype)
            np.testing.assert_equal(val, mlval)


@pytest.mark.parametrize("fi,mldtype", test_formats)
def test_round_ints(fi, mldtype):
    for v in np.arange(289).astype(float):
        val = round_float(fi, v)

        mlval = mlround(v, mldtype)
        np.testing.assert_equal(val, mlval)

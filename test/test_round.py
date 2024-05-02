# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from typing import Type

import ml_dtypes
import numpy as np
import pytest

from gfloat import RoundMode, decode_float, round_float
from gfloat.formats import *


def _mlround(v: float, dty: Type) -> float:
    """
    Round `v` using ml_dtypes library
    """
    return np.array([v]).astype(dty).astype(float).item()


def test_round_p3109() -> None:
    fi = format_info_p3109(4)
    assert round_float(fi, 0.0068359375) == 0.0068359375
    assert round_float(fi, 0.0029296875) == 0.0029296875
    assert round_float(fi, 0.0078125) == 0.0078125
    assert round_float(fi, 0.017578125) == 0.017578125
    assert round_float(fi, 224.0) == 224.0
    assert round_float(fi, 240.0) == np.inf

    assert round_float(fi, 224.1, RoundMode.TowardPositive) == np.inf

    assert round_float(fi, 232.0) == 224.0
    assert round_float(fi, 232.0, RoundMode.TiesToAway) == np.inf
    assert round_float(fi, 232.0, RoundMode.TowardZero) == 224.0
    assert round_float(fi, 232.0, RoundMode.TowardNegative) == 224.0
    assert round_float(fi, 232.0, RoundMode.TowardPositive) == np.inf

    assert round_float(fi, -232.0) == -224.0
    assert round_float(fi, -232.0, RoundMode.TiesToAway) == -np.inf
    assert round_float(fi, -232.0, RoundMode.TowardZero) == -224.0
    assert round_float(fi, -232.0, RoundMode.TowardNegative) == -np.inf
    assert round_float(fi, -232.0, RoundMode.TowardPositive) == -224.0

    assert round_float(fi, 232.1) == np.inf


def test_round_e5m2() -> None:
    fi = format_info_ocp_e5m2

    assert fi.max == 57344

    assert round_float(fi, 1.5258789e-05) == 2**-16

    # Default NONSAT rounding
    assert round_float(fi, 57344.0) == 57344
    assert round_float(fi, 57344.1) == 57344
    assert round_float(fi, 61439.9) == 57344
    assert round_float(fi, 61440.0) == np.inf
    assert round_float(fi, np.inf, sat=False) == np.inf
    assert round_float(fi, -np.inf, sat=False) == -np.inf
    assert np.isnan(round_float(fi, np.nan, sat=False))

    # SAT rounding
    assert round_float(fi, 57344.0, sat=True) == 57344
    assert round_float(fi, 57344.1, sat=True) == 57344
    assert round_float(fi, 61439.9, sat=True) == 57344
    assert round_float(fi, 61440.0, sat=True) == 57344
    assert round_float(fi, np.inf, sat=True) == 57344
    assert round_float(fi, -np.inf, sat=True) == -57344
    assert np.isnan(round_float(fi, np.nan, sat=True))


def test_round_e4m3() -> None:
    fi = format_info_ocp_e4m3

    assert fi.max == 448

    # Default NONSAT rounding
    assert round_float(fi, 448.0) == 448
    assert round_float(fi, 448.1) == 448
    assert round_float(fi, 464.0) == 448
    assert np.isnan(round_float(fi, 464.01))
    assert np.isnan(round_float(fi, np.inf, sat=False))
    assert np.isnan(round_float(fi, -np.inf, sat=False))
    assert np.isnan(round_float(fi, np.nan, sat=False))

    # SAT rounding
    assert round_float(fi, 448.0, sat=True) == 448
    assert round_float(fi, 448.1, sat=True) == 448
    assert round_float(fi, 464.0, sat=True) == 448
    assert round_float(fi, 464.01, sat=True) == 448
    assert round_float(fi, np.inf, sat=True) == 448
    assert round_float(fi, -np.inf, sat=True) == -448
    assert np.isnan(round_float(fi, np.nan, sat=True))


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


@pytest.mark.parametrize(
    "fi",
    [
        format_info_ocp_e5m2,
        format_info_ocp_e4m3,
        *p3109_formats,
    ],
    ids=str,
)
def test_round(fi: FormatInfo) -> None:
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
    for i in some_positive_codepoints:
        v0 = decode_float(fi, i + 0).fval
        v1 = decode_float(fi, i + 1).fval
        if np.isfinite([v0, v1]).all():
            dv = v1 - v0
            np.testing.assert_equal(round_float(fi, v0), v0)
            np.testing.assert_equal(round_float(fi, v0 + 0.3 * dv), v0)
            np.testing.assert_equal(round_float(fi, v0 + 0.49 * dv), v0)
            np.testing.assert_equal(round_float(fi, v0 + 0.51 * dv), v1)
            np.testing.assert_equal(round_float(fi, v0 + 0.99 * dv), v1)
            nearest_even = v0 if (i & 1 == 0) else v1
            np.testing.assert_equal(round_float(fi, v0 + 0.50 * dv), nearest_even)


test_formats = [
    (format_info_ocp_e5m2, ml_dtypes.float8_e5m2),
    (format_info_ocp_e4m3, ml_dtypes.float8_e4m3fn),
]


def _linterp(a: float, b: float, t: float) -> float:
    return a * (1 - t) + b * t


@pytest.mark.parametrize("fi,mldtype", test_formats)
def test_ml_dtype_compatible(fi: FormatInfo, mldtype: Type) -> None:
    """
    Test that rounding is compatible with ml_dtypes, assuming IEEE-like rounding
    """
    for i in range(255):
        # For each float v, check values at various interpolations
        # between v and nextUp(v)
        v0 = decode_float(fi, i + 0).fval
        v1 = decode_float(fi, i + 1).fval

        for alpha in (0, 0.3, 0.5, 0.6, 0.9, 1.25):
            v = _linterp(v0, v1, alpha)
            if np.isfinite(v):
                val = round_float(fi, v, RoundMode.TiesToEven)

                mlval = _mlround(v, mldtype)
                np.testing.assert_equal(val, mlval)


@pytest.mark.parametrize("fi,mldtype", test_formats)
def test_round_ints(fi: FormatInfo, mldtype: Type) -> None:
    for v in np.arange(289).astype(float):
        val = round_float(fi, v)

        mlval = _mlround(v, mldtype)
        np.testing.assert_equal(val, mlval)


@pytest.mark.parametrize("fi", all_formats, ids=str)
def test_round_roundtrip(fi: FormatInfo) -> None:
    if fi.bits <= 8:
        step = 1
    elif fi.bits <= 16:
        step = 13
    elif fi.bits <= 32:
        step = 73013

    for i in range(0, 2**fi.bits, step):
        fv = decode_float(fi, i)
        fval2 = round_float(fi, fv.fval)
        np.testing.assert_equal(fval2, fv.fval)

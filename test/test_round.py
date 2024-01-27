import pytest
import ml_dtypes
import numpy as np

from gfloat import decode_float, round_float
from gfloat.formats import *

test_formats = [
    (format_info_ocp_e5m2, ml_dtypes.float8_e5m2),
    (format_info_ocp_e4m3, ml_dtypes.float8_e4m3fn),
]


def mlround(v, dty):
    return np.array([v]).astype(dty).astype(float).item()


def linterp(a, b, t):
    return a * (1 - t) + b * t


@pytest.mark.parametrize("i", (0, 1, 2, 3, 7, 15, 23, 35, 47, 53, 123, 126, 127))
@pytest.mark.parametrize("fi,mldtype", test_formats)
def test_round(fi, mldtype, i):
    v0 = decode_float(fi, i + 0).fval
    v1 = decode_float(fi, i + 1).fval
    for alpha in np.arange(0, 1, 0.3):
        v = linterp(v0, v1, alpha)

        val = round_float(fi, v)

        mlval = mlround(v, mldtype)
        print(f"**\n{v=}, {val=}, {mlval=}")
        np.testing.assert_equal(val, mlval)


@pytest.mark.parametrize("fi,mldtype", test_formats)
def test_round_ints(fi, mldtype):
    for v in np.arange(256).astype(float):
        val = round_float(fi, v)

        mlval = mlround(v, mldtype)
        print(f"**\n{v=}, {val=}, {mlval=}")
        np.testing.assert_equal(val, mlval)

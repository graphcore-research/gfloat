# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import pytest
import numpy as np
import ml_dtypes
from gfloat import decode_float, FloatClass
from gfloat.formats import *


def test_spot_check_ocp_e5m2():
    fi = format_info_ocp_e5m2
    dec = lambda ival: decode_float(fi, ival).fval
    fclass = lambda ival: decode_float(fi, ival).fclass
    assert dec(0x01) == 2.0**-16
    assert dec(0x40) == 2.0
    assert dec(0x80) == 0.0 and np.signbit(dec(0x80))
    assert dec(0x7B) == 57344.0
    assert dec(0x7C) == np.inf
    assert np.floor(np.log2(dec(0x7B))) == fi.emax
    assert dec(0xFC) == -np.inf
    assert np.isnan(dec(0x7F))
    assert fclass(0x80) == FloatClass.ZERO
    assert fclass(0x00) == FloatClass.ZERO


def test_spot_check_ocp_e4m3():
    fi = format_info_ocp_e4m3
    dec = lambda ival: decode_float(fi, ival).fval
    fclass = lambda ival: decode_float(fi, ival).fclass
    assert dec(0x40) == 2.0
    assert dec(0x01) == 2.0**-9
    assert dec(0x80) == 0.0 and np.signbit(dec(0x80))
    assert np.isnan(dec(0x7F))
    assert dec(0x7E) == 448.0
    assert np.floor(np.log2(dec(0x7E))) == fi.emax


def test_spot_check_p3109_p3():
    fi = format_info_p3109(3)
    dec = lambda ival: decode_float(fi, ival).fval
    fclass = lambda ival: decode_float(fi, ival).fclass
    assert dec(0x01) == 2.0**-17
    assert dec(0x40) == 1.0
    assert np.isnan(dec(0x80))
    assert dec(0xFF) == -np.inf
    assert np.floor(np.log2(dec(0x7E))) == fi.emax


def test_spot_check_p3109_p1():
    fi = format_info_p3109(1)
    dec = lambda ival: decode_float(fi, ival).fval
    fclass = lambda ival: decode_float(fi, ival).fclass
    assert dec(0x01) == 2.0**-62
    assert dec(0x40) == 2.0
    assert np.isnan(dec(0x80))
    assert dec(0xFF) == -np.inf
    assert np.floor(np.log2(dec(0x7E))) == fi.emax


def test_spot_check_binary16():
    fi = format_info_binary16
    dec = lambda ival: decode_float(fi, ival).fval
    fclass = lambda ival: decode_float(fi, ival).fclass
    assert dec(0x3C00) == 1.0
    assert dec(0x3C01) == 1.0 + 2**-10
    assert dec(0x4000) == 2.0
    assert dec(0x0001) == 2**-24
    assert dec(0x7BFF) == 65504.0
    assert np.isinf(dec(0x7C00))
    assert np.isnan(dec(0x7C01))
    assert np.isnan(dec(0x7FFF))


def test_spot_check_bfloat16():
    fi = format_info_bfloat16
    dec = lambda ival: decode_float(fi, ival).fval
    fclass = lambda ival: decode_float(fi, ival).fclass
    assert dec(0x3F80) == 1
    assert dec(0x4000) == 2
    assert dec(0x0001) == 2**-133
    assert dec(0x4780) == 65536.0
    assert np.isinf(dec(0x7F80))
    assert np.isnan(dec(0x7F81))
    assert np.isnan(dec(0x7FFF))


@pytest.mark.parametrize(
    "fmt,npfmt,int_dtype",
    [
        (format_info_binary16, np.float16, np.uint16),
        (format_info_bfloat16, ml_dtypes.bfloat16, np.uint16),
        (format_info_ocp_e4m3, ml_dtypes.float8_e4m3fn, np.uint8),
    ],
)
def test_consistent_decodes_all_values(fmt, npfmt, int_dtype):
    npivals = np.arange(
        np.iinfo(int_dtype).min, int(np.iinfo(int_dtype).max) + 1, dtype=int_dtype
    )
    npfvals = npivals.view(dtype=npfmt)
    for i, npfval in zip(npivals, npfvals):
        val = decode_float(fmt, int(i))
        np.testing.assert_equal(val.fval, npfval)


@pytest.mark.parametrize("v", [-1, 0x10000])
def test_except(v):
    with pytest.raises(ValueError):
        decode_float(format_info_binary16, v)

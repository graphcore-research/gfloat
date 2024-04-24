# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import ml_dtypes
import numpy as np
import pytest

from gfloat import FloatClass, decode_float
from gfloat.formats import *


def _isnegzero(x):
    return (x == 0) and (np.signbit(x) == 1)


def test_spot_check_ocp_e5m2():
    fi = format_info_ocp_e5m2
    dec = lambda ival: decode_float(fi, ival).fval
    fclass = lambda ival: decode_float(fi, ival).fclass
    assert dec(0x01) == 2.0**-16
    assert dec(0x40) == 2.0
    assert _isnegzero(dec(0x80))
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
    assert _isnegzero(dec(0x80))
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


def test_spot_check_ocp_e2m3():
    # Test against Table 4 in "OCP Microscaling Formats (MX) v1.0 Spec"
    fi = format_info_ocp_e2m3
    dec = lambda ival: decode_float(fi, ival).fval
    fclass = lambda ival: decode_float(fi, ival).fclass
    assert fi.max == 7.5
    assert fi.smallest_subnormal == 0.125
    assert fi.smallest_normal == 1.0

    assert dec(0b000000) == 0
    assert dec(0b011111) == 7.5
    assert _isnegzero(dec(0b100000))


def test_spot_check_ocp_e3m2():
    # Test against Table 4 in "OCP Microscaling Formats (MX) v1.0 Spec"
    fi = format_info_ocp_e3m2
    dec = lambda ival: decode_float(fi, ival).fval
    fclass = lambda ival: decode_float(fi, ival).fclass
    assert fi.max == 28.0
    assert fi.smallest_subnormal == 0.0625
    assert fi.smallest_normal == 0.25

    assert dec(0b000000) == 0
    assert dec(0b011111) == 28.0
    assert _isnegzero(dec(0b100000))


def test_spot_check_ocp_e2m1():
    # Test against Table 5 in "OCP Microscaling Formats (MX) v1.0 Spec"
    fi = format_info_ocp_e2m1
    dec = lambda ival: decode_float(fi, ival).fval
    fclass = lambda ival: decode_float(fi, ival).fclass
    assert fi.max == 6.0
    assert fi.smallest_subnormal == 0.5
    assert fi.smallest_normal == 1.0

    assert dec(0b0000) == 0
    assert dec(0b0001) == 0.5
    assert dec(0b0010) == 1.0
    assert dec(0b0011) == 1.5
    assert dec(0b0100) == 2.0
    assert dec(0b0101) == 3.0
    assert dec(0b0110) == 4.0
    assert dec(0b0111) == 6.0
    assert _isnegzero(dec(0b1000))


def test_spot_check_ocp_e8m0():
    fi = format_info_ocp_e8m0
    dec = lambda ival: decode_float(fi, ival).fval
    fclass = lambda ival: decode_float(fi, ival).fclass
    assert dec(0x00) == 2.0**-127
    assert dec(0x01) == 2.0**-126
    assert dec(0x7F) == 1.0
    assert np.isnan(dec(0xFF))
    assert fclass(0x80) == FloatClass.NORMAL
    assert fclass(0x00) == FloatClass.NORMAL
    assert fi.max == 2.0**127


@pytest.mark.parametrize("fi", p3109_formats, ids=str)
def test_specials(fi):
    assert fi.code_of_nan == 0x80
    assert fi.code_of_zero == 0x00
    assert fi.code_of_posinf == 0x7F
    assert fi.code_of_neginf == 0xFF


@pytest.mark.parametrize("fi", all_formats, ids=str)
def test_specials_decode(fi):
    dec = lambda v: decode_float(fi, v).fval

    assert dec(fi.code_of_zero) == 0

    if fi.num_nans > 0:
        assert np.isnan(dec(fi.code_of_nan))

    if fi.has_infs:
        assert dec(fi.code_of_posinf) == np.inf
        assert dec(fi.code_of_neginf) == -np.inf

    assert dec(fi.code_of_max) == fi.max
    assert dec(fi.code_of_min) == fi.min

    assert dec(1) == fi.smallest


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

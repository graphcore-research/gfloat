# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from typing import Callable

import numpy as np
import numpy.typing as npt
import pytest

from gfloat import decode_float, encode_float, encode_ndarray
from gfloat.formats import *


@pytest.mark.parametrize("fi", sample_formats)
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
        assert (i == code) or (np.isnan(fv.fval) and code == fi.code_of_nan)
        fv2 = decode_float(fi, code)
        np.testing.assert_equal(fv2.fval, fv.fval)

    codes = np.arange(0, 2**fi.bits, step, dtype=np.uint64)
    fvals = np.array([decode_float(fi, int(i)).fval for i in codes])
    enc_codes = encode_ndarray(fi, fvals)
    expected_codes: npt.NDArray
    if fi.num_nans == 0:
        assert not np.any(np.isnan(fvals))
        expected_codes = codes
    else:
        expected_codes = np.where(np.isnan(fvals), fi.code_of_nan, codes)
    np.testing.assert_equal(enc_codes, expected_codes)


@pytest.mark.parametrize("fi", sample_formats)
@pytest.mark.parametrize("enc", (encode_float, encode_ndarray))
def test_encode_edges(fi: FormatInfo, enc: Callable) -> None:
    if enc == encode_ndarray:
        enc = lambda fi, x: encode_ndarray(fi, np.array([x])).item()

    assert enc(fi, fi.max) == fi.code_of_max

    assert enc(fi, fi.max * 1.25) == (
        fi.code_of_posinf
        if fi.domain == Domain.Extended
        else fi.code_of_nan if fi.num_nans > 0 else fi.code_of_max
    )

    if fi.is_signed:
        assert enc(fi, fi.min * 1.25) == (
            fi.code_of_neginf
            if fi.domain == Domain.Extended
            else fi.code_of_nan if fi.num_nans > 0 else fi.code_of_min
        )


@pytest.mark.parametrize(
    "values",
    [
        [0.0, -0.0],
        [[-0.0, 0.0], [1.0, -1.0]],
        [1.0, -1.0],  # The zero selection is empty, as in issue #63.
        [],
    ],
)
def test_encode_binary64_signed_zero(values: npt.ArrayLike) -> None:
    v = np.array(values, dtype=np.float64)
    codes = encode_ndarray(format_info_binary64, v)
    assert codes.dtype == np.dtype(np.uint64)
    assert codes.shape == v.shape
    np.testing.assert_array_equal(codes, v.view(np.uint64))


@pytest.mark.parametrize("bias", [-3, 0, 3])
@pytest.mark.parametrize("precision", [2, 3])
@pytest.mark.parametrize(
    "signed, negative_zero", [(False, False), (True, False), (True, True)]
)
def test_encode_zero_without_subnormals(
    bias: int, precision: int, signed: bool, negative_zero: bool
) -> None:
    fi = FormatInfo(
        name="no_subnormals",
        k=5,
        precision=precision,
        bias=bias,
        is_signed=signed,
        domain=Domain.Finite,
        has_nz=negative_zero,
        num_high_nans=0,
        has_subnormals=False,
        is_twos_complement=False,
    )
    values = np.array([0.0, -0.0])
    expected = [fi.code_of_zero, fi.code_of_negzero if negative_zero else fi.code_of_zero]
    assert [encode_float(fi, value) for value in values] == expected
    np.testing.assert_array_equal(encode_ndarray(fi, values), expected)
    for code, value in zip(expected, values):
        decoded = decode_float(fi, code).fval
        assert decoded == 0.0
        assert np.signbit(decoded) == (negative_zero and np.signbit(value))

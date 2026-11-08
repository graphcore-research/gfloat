# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from types import ModuleType

import array_api_strict as xp
import numpy as np
import pytest

from gfloat import (
    RoundMode,
    FormatInfo,
    decode_float,
    decode_ndarray,
    round_float,
    round_ndarray,
)
from gfloat.formats import *
from gfloat.round_ndarray import _frexp

xp.set_array_api_strict_flags(api_version="2024.12")


@pytest.mark.parametrize("fi", sample_formats)
@pytest.mark.parametrize("rnd", RoundMode)
@pytest.mark.parametrize("sat", [True, False])
def test_array_api(fi: FormatInfo, rnd: RoundMode, sat: bool) -> None:
    a0 = np.random.rand(23, 1, 34) - 0.5
    a = xp.asarray(a0)

    srnumbits = 32
    srbits0 = np.random.randint(0, 2**srnumbits, a.shape, dtype=np.int64)
    srbits = xp.asarray(srbits0)

    round_ndarray(fi, a, rnd, sat, srbits=srbits, srnumbits=srnumbits)  # type: ignore


@pytest.mark.parametrize("namespace", (np, xp), ids=("numpy", "strict"))
@pytest.mark.parametrize("dtype_name", ("float32", "float64"))
def test_frexp_exponents(namespace: ModuleType, dtype_name: str) -> None:
    dtype = np.dtype(dtype_name)
    info = np.finfo(dtype)
    exponents = [-126, -100, -2, 0, 1, 10, 100]
    if dtype_name == "float64":
        exponents += [-1022, -1000, 1000, 1023]
    values = [
        float(info.smallest_subnormal),
        float(info.smallest_normal),
        float(info.max),
    ]
    for exponent in exponents:
        value = np.asarray(2.0**exponent, dtype=dtype)
        values += [
            float(np.nextafter(value, np.asarray(0.0, dtype=dtype))),
            float(value),
            float(np.nextafter(value, np.asarray(np.inf, dtype=dtype))),
        ]
    array = np.asarray(values, dtype=dtype).reshape(1, -1)
    expected = np.frexp(array)[1]
    with np.errstate(over="raise", invalid="raise"):
        actual = _frexp(namespace.asarray(array))[1]
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize(
    "fi", (format_info_ocp_e2m1, format_info_ocp_e2m3, format_info_ocp_e3m2)
)
@pytest.mark.parametrize("rnd", RoundMode)
@pytest.mark.parametrize("sat", (False, True))
@pytest.mark.parametrize("dtype_name", ("float32", "float64"))
def test_array_api_rounding_agrees(
    fi: FormatInfo, rnd: RoundMode, sat: bool, dtype_name: str
) -> None:
    values = np.asarray(
        [
            -5.75,
            -5.0,
            -3.25,
            -2.75,
            -2.5,
            -1.75,
            -1.25,
            -0.75,
            -0.5,
            -0.25,
            -0.0,
            0.0,
            0.25,
            0.5,
            0.75,
            1.25,
            1.75,
            2.5,
            2.75,
            3.25,
            5.0,
            5.75,
            2.5,
            2.5,
        ],
        dtype=dtype_name,
    ).reshape(2, -1)[:, ::-1]
    original = values.copy()
    bits = np.resize(np.asarray([0, 1, 127, 128, 254, 255]), values.shape)
    expected = np.asarray(
        [
            round_float(fi, float(value), rnd, sat, srbits=int(bit), srnumbits=8)
            for value, bit in zip(values.flat, bits.flat)
        ]
    ).reshape(values.shape)
    native = round_ndarray(fi, values, rnd, sat, srbits=bits, srnumbits=8)
    strict = round_ndarray(
        fi, xp.asarray(values), rnd, sat, srbits=xp.asarray(bits), srnumbits=8  # type: ignore[arg-type]
    )
    for actual in (native, np.asarray(strict)):
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(np.signbit(values), np.signbit(original))


@pytest.mark.parametrize("dtype_name", ("float32", "float64"))
def test_array_api_empty(dtype_name: str) -> None:
    values = xp.asarray(np.empty((2, 0), dtype=dtype_name))
    actual = round_ndarray(format_info_ocp_e2m1, values)  # type: ignore[arg-type]
    assert actual.shape == values.shape

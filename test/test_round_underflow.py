# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from typing import Callable

import numpy as np
import pytest

from gfloat import FormatInfo, RoundMode, decode_float, encode_float
from gfloat import round_float, round_ndarray
from gfloat.formats import Domain, format_info_ocp_e8m0


def scalar_round(
    fi: FormatInfo, value: float, rnd: RoundMode, sat: bool, bits: int
) -> float:
    return round_float(fi, value, rnd, sat, srbits=bits, srnumbits=8)


def array_round(
    fi: FormatInfo, value: float, rnd: RoundMode, sat: bool, bits: int
) -> float:
    return float(
        round_ndarray(
            fi, np.array([value]), rnd, sat, srbits=np.array([bits]), srnumbits=8
        )[0]
    )


def no_subnormal_format(precision: int, signed: bool = False) -> FormatInfo:
    return FormatInfo(
        name="no_subnormals",
        k=5,
        precision=precision,
        bias=3,
        is_signed=signed,
        domain=Domain.Finite,
        has_nz=signed,
        num_high_nans=0,
        has_subnormals=False,
        is_twos_complement=False,
    )


@pytest.mark.parametrize("rounder", [scalar_round, array_round])
@pytest.mark.parametrize("rnd", RoundMode)
@pytest.mark.parametrize("sat", [False, True])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("bits", [0, 255])
def test_round_no_subnormals_zero_gap(
    rounder: Callable, rnd: RoundMode, sat: bool, sign: int, fraction: float, bits: int
) -> None:
    fi = no_subnormal_format(precision=2, signed=True)
    if rnd == RoundMode.TowardZero:
        up = False
    elif rnd == RoundMode.TowardPositive:
        up = sign > 0
    elif rnd == RoundMode.TowardNegative:
        up = sign < 0
    elif rnd == RoundMode.TiesToEven:
        up = fraction > 0.5
    elif rnd == RoundMode.TiesToAway:
        up = fraction >= 0.5
    elif rnd == RoundMode.ToOdd:
        up = True
    else:
        up = bits == 255
    expected = sign * (fi.smallest if up else 0.0)
    actual = rounder(fi, sign * fraction * fi.smallest, rnd, sat, bits)
    assert actual == expected
    assert np.signbit(actual) == np.signbit(expected)
    assert actual in [decode_float(fi, code).fval for code in range(2**fi.bits)]
    assert decode_float(fi, encode_float(fi, actual)).fval == actual


@pytest.mark.parametrize("rounder", [scalar_round, array_round])
@pytest.mark.parametrize("rnd", RoundMode)
@pytest.mark.parametrize("sat", [False, True])
@pytest.mark.parametrize("bits", [0, 255])
@pytest.mark.parametrize(
    "fi", [format_info_ocp_e8m0, no_subnormal_format(1), no_subnormal_format(1, True)]
)
def test_round_no_zero_underflow(
    rounder: Callable, rnd: RoundMode, sat: bool, bits: int, fi: FormatInfo
) -> None:
    values = [
        0.0,
        -0.0,
        np.nextafter(0.0, 1.0),
        fi.smallest / 8,
        fi.smallest / 2,
        0.75 * fi.smallest,
        np.nextafter(fi.smallest, 0.0),
        fi.smallest,
    ]
    if fi.is_signed:
        values += [-v for v in values]
    for value in values:
        actual = rounder(fi, value, rnd, sat, bits)
        expected = -fi.smallest if fi.is_signed and np.signbit(value) else fi.smallest
        assert actual == expected
        assert decode_float(fi, encode_float(fi, actual)).fval == expected


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("sat", [False, True])
def test_round_e8m0_mixed_array(strict: bool, sat: bool) -> None:
    import array_api_strict

    xp = array_api_strict if strict else np
    fi = format_info_ocp_e8m0
    values = xp.asarray(
        [[0.0, fi.smallest / 2, fi.smallest], [1.0, np.inf, np.nan]], dtype=xp.float64
    )
    result = round_ndarray(fi, values, sat=sat)
    assert result.shape == values.shape
    assert result.dtype == values.dtype
    np.testing.assert_equal(
        np.asarray(result),
        [
            [fi.smallest, fi.smallest, fi.smallest],
            [1.0, fi.max if sat else np.nan, np.nan],
        ],
    )
    empty = xp.asarray([], dtype=xp.float64)
    assert round_ndarray(fi, empty, sat=sat).shape == (0,)


@pytest.mark.parametrize("rnd", RoundMode)
@pytest.mark.parametrize("bits", [0, 255])
def test_round_zero_gap_array_api(rnd: RoundMode, bits: int) -> None:
    import array_api_strict as xp

    fi = no_subnormal_format(precision=2, signed=True)
    values = np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]) * fi.smallest
    result = round_ndarray(
        fi, xp.asarray(values), rnd, srbits=xp.asarray([bits] * len(values)), srnumbits=8  # type: ignore[arg-type]
    )
    expected = [scalar_round(fi, v, rnd, False, bits) for v in values]
    np.testing.assert_equal(np.asarray(result), expected)
    np.testing.assert_array_equal(np.signbit(np.asarray(result)), np.signbit(expected))

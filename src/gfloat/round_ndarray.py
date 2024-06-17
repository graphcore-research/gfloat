# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from .types import FormatInfo, RoundMode
import numpy as np
import math


def _isodd(v: np.ndarray) -> np.ndarray:
    return v & 0x1 == 1


def round_ndarray(
    fi: FormatInfo,
    v: np.ndarray,
    rnd: RoundMode = RoundMode.TiesToEven,
    sat: bool = False,
    np=np,
) -> np.ndarray:
    """
    Vectorized version of round_float.
    """
    p = fi.precision
    bias = fi.expBias

    result = np.zeros_like(v)
    is_negative = np.signbit(v) & fi.is_signed
    vpos = np.where(is_negative, -v, v)

    nan_mask = np.isnan(v)
    result[nan_mask] = np.nan

    inf_mask = np.isinf(vpos)
    result[inf_mask] = np.inf

    zero_mask = vpos == 0
    result[zero_mask] = 0

    finite_mask = ~(nan_mask | inf_mask | zero_mask)

    if np.any(finite_mask):
        expval = np.floor(np.log2(vpos[finite_mask])).astype(int)

        if fi.has_subnormals:
            expval = np.maximum(expval, 1 - bias)

        expval = expval - p + 1
        fsignificand = np.ldexp(vpos[finite_mask], -expval)

        isignificand = np.floor(fsignificand).astype(np.int64)
        delta = fsignificand - isignificand

        if fi.precision > 1:
            code_is_odd = _isodd(isignificand)
        else:
            code_is_odd = (isignificand != 0) & _isodd(expval + bias)

        if rnd == RoundMode.TowardPositive:
            round_up = ~is_negative[finite_mask] & (delta > 0)
        elif rnd == RoundMode.TowardNegative:
            round_up = is_negative[finite_mask] & (delta > 0)
        elif rnd == RoundMode.TiesToAway:
            round_up = delta >= 0.5
        elif rnd == RoundMode.TiesToEven:
            round_up = (delta > 0.5) | ((delta == 0.5) & code_is_odd)
        else:
            round_up = np.zeros_like(delta, dtype=bool)

        if fi.precision > 1:
            isignificand[round_up] += 1
        else:
            # if isignificand == 0:
            #     isignificand = 1
            # else:
            #     assert isignificand == 1
            #     expval += 1
            expval[round_up & (isignificand == 1)] += 1
            isignificand[round_up] = 1

        result[finite_mask] = isignificand * (2.0**expval)

    amax = np.where(is_negative, -fi.min, fi.max)

    # In some rounding modes, send to amax independent of sat
    if rnd == RoundMode.TowardNegative:
        result[(result > amax) & finite_mask & ~is_negative] = amax
    elif rnd == RoundMode.TowardPositive:
        result[(result > amax) & finite_mask & is_negative] = amax
    elif rnd == RoundMode.TowardZero:
        result[(result > amax) & finite_mask] = amax

    if sat:
        result[result > amax] = amax[result > amax]
    else:
        # Not saturating, send to infinity or NaN
        if fi.has_infs:
            result[result > amax] = np.inf
        elif fi.num_nans > 0:
            result[result > amax] = np.nan
        else:
            if np.any(result > amax):
                raise ValueError(f"No Infs or NaNs in format {fi}, and sat=False")

    result[is_negative] = -result[is_negative]

    if np.any(result == 0):
        if fi.has_nz:
            result[(result == 0) & is_negative] = -0.0
        else:
            result[result == 0] = 0.0

    return result


def encode_ndarray(fi: FormatInfo, v: np.ndarray) -> np.ndarray:
    """
    Vectorized version of encode_float.
    """
    k = fi.bits
    p = fi.precision
    t = p - 1

    sign = np.signbit(v) & fi.is_signed
    vpos = np.where(sign, -v, v)

    nan_mask = np.isnan(v)
    inf_mask = np.isinf(v)

    code = np.zeros_like(v, dtype=np.uint64)

    if fi.num_nans > 0:
        code[nan_mask] = fi.code_of_nan
    else:
        assert not np.any(nan_mask)

    if fi.has_infs:
        code[v > fi.max] = fi.code_of_posinf
        code[v < fi.min] = fi.code_of_neginf
    else:
        code[v > fi.max] = fi.code_of_nan if fi.num_nans > 0 else fi.code_of_max
        code[v < fi.min] = fi.code_of_nan if fi.num_nans > 0 else fi.code_of_min

    if fi.has_zero:
        if fi.has_nz:
            code[v == 0] = np.where(sign[v == 0], fi.code_of_negzero, fi.code_of_zero)
        else:
            code[v == 0] = fi.code_of_zero

    finite_mask = (code == 0) & (v != 0)
    assert not np.any(np.isnan(vpos[finite_mask]))
    if np.any(finite_mask):
        finite_vpos = vpos[finite_mask]
        finite_sign = sign[finite_mask]

        sig, exp = np.frexp(finite_vpos)
        expl = exp.astype(int) - 1
        tsig = sig * 2

        biased_exp = expl + fi.expBias
        subnormal_mask = (biased_exp < 1) & fi.has_subnormals

        tsig[subnormal_mask] *= 2.0 ** (biased_exp[subnormal_mask] - 1)
        biased_exp[subnormal_mask] = 0
        tsig[~subnormal_mask] -= 1.0

        isig = np.floor(tsig * 2**t).astype(int)

        zero_mask = fi.has_zero & (isig == 0) & (biased_exp == 0)
        if not fi.has_nz:
            finite_sign[zero_mask] = False

        # Handle two's complement encoding
        if fi.is_twos_complement:
            isig[finite_sign] = (1 << t) - isig[finite_sign]

        code[finite_mask] = (
            (finite_sign.astype(int) << (k - 1)) | (biased_exp << t) | (isig << 0)
        )

    return code

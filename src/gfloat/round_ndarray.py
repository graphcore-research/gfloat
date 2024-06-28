# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from typing import Optional
from types import ModuleType
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
    srbits: Optional[np.ndarray] = None,
    srnumbits: int = 0,
    np: ModuleType = np,
) -> np.ndarray:
    """
    Vectorized version of :meth:`round_float`.

    Round inputs to the given :py:class:`FormatInfo`, given rounding mode and
    saturation flag

    Input NaNs will convert to NaNs in the target, not necessarily preserving payload.
    An input Infinity will convert to the largest float if :paramref:`sat`,
    otherwise to an Inf, if present, otherwise to a NaN.
    Negative zero will be returned if the format has negative zero, otherwise zero.

    Args:
      fi (FormatInfo): Describes the target format
      v (float): Input value to be rounded
      rnd (RoundMode): Rounding mode to use
      sat (bool): Saturation flag: if True, round overflowed values to `fi.max`
      np (Module): May be `numpy`, `jax.numpy` or another module cloning numpy

    Returns:
      An array of floats which is a subset of the format's value set.

    Raises:
       ValueError: The target format cannot represent an input
             (e.g. converting a `NaN`, or an `Inf` when the target has no
             `NaN` or `Inf`, and :paramref:`sat` is false)
    """
    p = fi.precision
    bias = fi.expBias

    is_negative = np.signbit(v) & fi.is_signed
    absv = np.where(is_negative, -v, v)

    finite_nonzero = ~(np.isnan(v) | np.isinf(v) | (v == 0))

    # Place 1.0 where finite_nonzero is False, to avoid log of {0,inf,nan}
    absv_masked = np.where(finite_nonzero, absv, 1.0)

    expval = np.floor(np.log2(absv_masked)).astype(int)

    if fi.has_subnormals:
        expval = np.maximum(expval, 1 - bias)

    expval = expval - p + 1
    fsignificand = np.ldexp(absv_masked, -expval)

    isignificand = np.floor(fsignificand).astype(np.int64)
    delta = fsignificand - isignificand

    if fi.precision > 1:
        code_is_odd = _isodd(isignificand)
    else:
        code_is_odd = (isignificand != 0) & _isodd(expval + bias)

    if rnd == RoundMode.TowardZero:
        should_round_away = np.zeros_like(delta, dtype=bool)
    if rnd == RoundMode.TowardPositive:
        should_round_away = ~is_negative & (delta > 0)
    if rnd == RoundMode.TowardNegative:
        should_round_away = is_negative & (delta > 0)
    if rnd == RoundMode.TiesToAway:
        should_round_away = delta >= 0.5
    if rnd == RoundMode.TiesToEven:
        should_round_away = (delta > 0.5) | ((delta == 0.5) & code_is_odd)
    if rnd == RoundMode.Stochastic:
        assert srbits is not None
        should_round_away = delta > (0.5 + srbits) * 2.0**-srnumbits
    if rnd == RoundMode.StochasticFast:
        assert srbits is not None
        should_round_away = delta > srbits * 2.0**-srnumbits

    isignificand = np.where(should_round_away, isignificand + 1, isignificand)

    result = np.where(finite_nonzero, np.ldexp(isignificand, expval), absv)

    amax = np.where(is_negative, -fi.min, fi.max)

    if sat:
        result = np.where(result > amax, amax, result)
    else:
        if rnd == RoundMode.TowardNegative:
            put_amax_at = (result > amax) & ~is_negative
        elif rnd == RoundMode.TowardPositive:
            put_amax_at = (result > amax) & is_negative
        elif rnd == RoundMode.TowardZero:
            put_amax_at = result > amax
        else:
            put_amax_at = np.zeros_like(result, dtype=bool)

        result = np.where(finite_nonzero & put_amax_at, amax, result)

        # Now anything larger than amax goes to infinity or NaN
        if fi.has_infs:
            result = np.where(result > amax, np.inf, result)
        elif fi.num_nans > 0:
            result = np.where(result > amax, np.nan, result)
        else:
            if np.any(result > amax):
                raise ValueError(f"No Infs or NaNs in format {fi}, and sat=False")

    result = np.where(is_negative, -result, result)

    # Make negative zeros negative if has_nz, else make them not negative.
    if fi.has_nz:
        result = np.where((result == 0) & is_negative, -0.0, result)
    else:
        result = np.where(result == 0, 0.0, result)

    return result


def encode_ndarray(fi: FormatInfo, v: np.ndarray) -> np.ndarray:
    """
    Vectorized version of :meth:`encode_float`.

    Encode inputs to the given :py:class:`FormatInfo`.

    Will round toward zero if :paramref:`v` is not in the value set.
    Will saturate to `Inf`, `NaN`, `fi.max` in order of precedence.
    Encode -0 to 0 if not `fi.has_nz`

    For other roundings and saturations, call :func:`round_ndarray` first.

    Args:
      fi (FormatInfo): Describes the target format
      v (float array): The value to be encoded.

    Returns:
      The integer code point
    """
    k = fi.bits
    p = fi.precision
    t = p - 1

    sign = np.signbit(v) & fi.is_signed
    vpos = np.where(sign, -v, v)

    nan_mask = np.isnan(v)

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

        biased_exp = exp.astype(np.int64) + (fi.expBias - 1)
        subnormal_mask = (biased_exp < 1) & fi.has_subnormals

        biased_exp_safe = np.where(subnormal_mask, biased_exp, 0)
        tsig = np.where(subnormal_mask, np.ldexp(sig, biased_exp_safe), sig * 2 - 1.0)
        biased_exp[subnormal_mask] = 0

        isig = np.floor(np.ldexp(tsig, t)).astype(np.int64)

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

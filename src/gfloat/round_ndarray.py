# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from typing import Optional
from types import ModuleType
from .types import FormatInfo, RoundMode
import numpy as np


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
      v (float array): Input values to be rounded
      rnd (RoundMode): Rounding mode to use
      sat (bool): Saturation flag: if True, round overflowed values to `fi.max`
      srbits (int array): Bits to use for stochastic rounding if rnd == Stochastic.
      srnumbits (int): How many bits are in srbits.  Implies srbits < 2**srnumbits.

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

    match rnd:
        case RoundMode.TowardZero:
            should_round_away = np.zeros_like(delta, dtype=bool)
        case RoundMode.TowardPositive:
            should_round_away = ~is_negative & (delta > 0)
        case RoundMode.TowardNegative:
            should_round_away = is_negative & (delta > 0)
        case RoundMode.TiesToAway:
            should_round_away = delta >= 0.5
        case RoundMode.TiesToEven:
            should_round_away = (delta > 0.5) | ((delta == 0.5) & code_is_odd)
        case RoundMode.Stochastic:
            assert srbits is not None
            ## RTNE delta to srbits
            d = delta * 2.0**srnumbits
            floord = np.floor(d).astype(np.int64)
            dd = d - floord
            drnd = floord + (dd > 0.5) + ((dd == 0.5) & _isodd(floord))

            should_round_away = drnd > srbits
        case RoundMode.StochasticOdd:
            assert srbits is not None
            ## RTNO delta to srbits
            d = delta * 2.0**srnumbits
            floord = np.floor(d).astype(np.int64)
            dd = d - floord
            drnd = floord + (dd > 0.5) + ((dd == 0.5) & ~_isodd(floord))

            should_round_away = drnd > srbits
        case RoundMode.StochasticFast:
            assert srbits is not None
            should_round_away = delta > (2 * srbits + 1) * 2.0 ** -(1 + srnumbits)
        case RoundMode.StochasticFastest:
            assert srbits is not None
            should_round_away = delta > srbits * 2.0**-srnumbits

    isignificand = np.where(should_round_away, isignificand + 1, isignificand)

    result = np.where(finite_nonzero, np.ldexp(isignificand, expval), absv)

    amax = np.where(is_negative, -fi.min, fi.max)

    if sat:
        result = np.where(result > amax, amax, result)
    else:
        match rnd:
            case RoundMode.TowardNegative:
                put_amax_at = (result > amax) & ~is_negative
            case RoundMode.TowardPositive:
                put_amax_at = (result > amax) & is_negative
            case RoundMode.TowardZero:
                put_amax_at = result > amax
            case _:
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

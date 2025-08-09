# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from typing import Optional
from types import ModuleType
from .types import FormatInfo, RoundMode

import numpy.typing as npt
import array_api_compat


def _isodd(v: npt.NDArray) -> npt.NDArray:
    return v & 0x1 == 1


def _ldexp(v: npt.NDArray, s: npt.NDArray) -> npt.NDArray:
    xp = array_api_compat.array_namespace(v, s)
    if (
        array_api_compat.is_torch_array(v)  # type: ignore
        or array_api_compat.is_jax_array(v)  # type: ignore
        or array_api_compat.is_numpy_array(v)
    ):
        return xp.ldexp(v, s)

    # Scale away from subnormal/infinite ranges
    offset = 24
    vlo = (v * 2.0**+offset) * 2.0 ** xp.astype(s - offset, v.dtype)
    vhi = (v * 2.0**-offset) * 2.0 ** xp.astype(s + offset, v.dtype)
    return xp.where(v < 1.0, vlo, vhi)


def _frexp(v: npt.NDArray) -> npt.NDArray:
    xp = array_api_compat.array_namespace(v)
    if (
        array_api_compat.is_torch_array(v)  # type: ignore
        or array_api_compat.is_jax_array(v)  # type: ignore
        or array_api_compat.is_numpy_array(v)
    ):
        return xp.frexp(v)

    raise NotImplementedError()


def round_ndarray(
    fi: FormatInfo,
    v: npt.NDArray,
    rnd: RoundMode = RoundMode.TiesToEven,
    sat: bool = False,
    srbits: Optional[npt.NDArray] = None,
    srnumbits: int = 0,
) -> npt.NDArray:
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

    Returns:
      An array of floats which is a subset of the format's value set.

    Raises:
       ValueError: The target format cannot represent an input
             (e.g. converting a `NaN`, or an `Inf` when the target has no
             `NaN` or `Inf`, and :paramref:`sat` is false)
    """
    xp = array_api_compat.array_namespace(v, srbits)

    # Until https://github.com/data-apis/array-api/issues/807
    xp_where = lambda a, t, f: xp.where(a, xp.asarray(t), xp.asarray(f))
    xp_maximum = lambda a, b: xp.maximum(xp.asarray(a), xp.asarray(b))

    p = fi.precision
    bias = fi.expBias

    is_negative = xp.signbit(v) & fi.is_signed
    absv = xp_where(is_negative, -v, v)

    finite_nonzero = ~(xp.isnan(v) | xp.isinf(v) | (v == 0))

    # Place 1.0 where finite_nonzero is False, to avoid log of {0,inf,nan}
    absv_masked = xp_where(finite_nonzero, absv, 1.0)

    int_type = xp.int64 if fi.k > 8 or srnumbits > 8 else xp.int16

    def to_int(x: npt.NDArray) -> npt.NDArray:
        return xp.astype(x, int_type)

    def to_float(x: npt.NDArray) -> npt.NDArray:
        return xp.astype(x, v.dtype)

    expval = _frexp(absv_masked)[1] - 1

    if fi.has_subnormals:
        expval = xp_maximum(expval, 1 - bias)

    expval = expval - p + 1
    fsignificand = _ldexp(absv_masked, -expval)

    floorfsignificand = xp.floor(fsignificand)
    isignificand = to_int(floorfsignificand)
    delta = fsignificand - floorfsignificand

    if fi.precision > 1:
        code_is_odd = _isodd(isignificand)
    else:
        code_is_odd = (isignificand != 0) & _isodd(expval + bias)

    match rnd:
        case RoundMode.TowardZero:
            should_round_away = xp.zeros_like(delta, dtype=xp.bool)

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
            d = delta * 2.0 ** float(srnumbits)
            floord = to_int(xp.floor(d))
            dd = d - xp.floor(d)
            should_round_away_tne = (dd > 0.5) | ((dd == 0.5) & _isodd(floord))
            drnd = floord + xp.astype(should_round_away_tne, floord.dtype)

            should_round_away = drnd + srbits >= int(2.0 ** float(srnumbits))

        case RoundMode.StochasticOdd:
            assert srbits is not None
            ## RTNO delta to srbits
            d = delta * 2.0 ** float(srnumbits)
            floord = to_int(xp.floor(d))
            dd = d - xp.floor(d)
            should_round_away_tno = (dd > 0.5) | ((dd == 0.5) & ~_isodd(floord))
            drnd = floord + xp.astype(should_round_away_tno, floord.dtype)

            should_round_away = drnd + srbits >= int(2.0 ** float(srnumbits))

        case RoundMode.StochasticFast:
            assert srbits is not None
            should_round_away = (
                delta + to_float(2 * srbits + 1) * 2.0 ** -float(1 + srnumbits) >= 1.0
            )

        case RoundMode.StochasticFastest:
            assert srbits is not None
            should_round_away = delta + to_float(srbits) * 2.0**-srnumbits >= 1.0

    isignificand = xp_where(should_round_away, isignificand + 1, isignificand)

    fresult = _ldexp(to_float(isignificand), expval)

    result = xp_where(finite_nonzero, fresult, absv)

    amax = xp_where(is_negative, -fi.min, fi.max)

    if sat:
        result = xp_where(result > amax, amax, result)
    else:
        match rnd:
            case RoundMode.TowardNegative:
                put_amax_at = (result > amax) & ~is_negative
            case RoundMode.TowardPositive:
                put_amax_at = (result > amax) & is_negative
            case RoundMode.TowardZero:
                put_amax_at = result > amax
            case _:
                put_amax_at = xp.zeros_like(result, dtype=xp.bool)

        result = xp_where(finite_nonzero & put_amax_at, amax, result)

        # Now anything larger than amax goes to infinity or NaN
        if fi.has_infs:
            result = xp_where(result > amax, xp.inf, result)
        elif fi.num_nans > 0:
            result = xp_where(result > amax, xp.nan, result)
        else:
            if xp.any(result > amax):
                raise ValueError(f"No Infs or NaNs in format {fi}, and sat=False")

    result = xp_where(is_negative, -result, result)

    # Make negative zeros negative if has_nz, else make them not negative.
    if fi.has_nz:
        result = xp_where((result == 0) & is_negative, -0.0, result)
    else:
        result = xp_where(result == 0, 0.0, result)

    return result

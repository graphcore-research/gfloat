# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from enum import Enum
import numpy as np
import math

from .types import FormatInfo, RoundMode


def _isodd(v: int):
    return v & 0x1 == 1


def round_float(fi: FormatInfo, v: float, rnd=RoundMode.TiesToEven, sat=False) -> float:
    """
    Round input to the given :py:class:`FormatInfo`, given rounding mode and saturation flag

    An input NaN will convert to a NaN in the target.
    An input Infinity will convert to the largest float if |sat|,
    otherwise to an Inf, if present, otherwise to a NaN.
    Negative zero will be returned if the format has negative zero.

    :param fi: FormatInfo describing the target format
    :type fi: FormatInfo

    :param v: Input float
    :type v: float

    :param rnd: Rounding mode to use
    :type rnd: RoundMode

    :param sat: Saturation flag: if True, round overflowed values to `fi.max`
    :type sat: bool

    :raises ValueError: The target format cannot represent the input
       (e.g. converting a NaN, or an Inf when the target has no Inf or NaN, and `¬sat`)

    :return: A float which equals (inc. nan) one of the values in the format
    :rtype: float
    """

    # Constants
    k = fi.k
    p = fi.precision
    w = fi.expBits
    bias = fi.expBias
    t = p - 1

    if np.isnan(v):
        if fi.num_nans == 0:
            raise ValueError(f"No NaN in format {fi}")

        # Note that this does not preserve the NaN payload
        return np.nan

    # Extract sign
    sign = np.signbit(v)

    if v == 0:
        result = 0

    elif np.isinf(v):
        result = np.inf

    else:
        # Extract significand (mantissa) and exponent
        fsignificand, expval = np.frexp(np.abs(v))

        assert fsignificand >= 0.5 and fsignificand < 1.0
        # move significand to [1.0, 2.0)
        fsignificand *= 2
        expval -= 1

        # Effective precision, accounting for right shift for subnormal values
        biased_exp = expval + bias
        effective_precision = t + min(biased_exp - 1, 0)

        # Lift to "integer * 2^e"
        fsignificand *= 2.0**effective_precision
        expval -= effective_precision

        # round
        isignificand = math.floor(fsignificand)
        if isignificand != fsignificand:
            # Need to round
            if rnd == RoundMode.TowardZero:
                pass
            elif rnd == RoundMode.TowardPositive:
                isignificand += 1 if not sign else 0
            elif rnd == RoundMode.TowardNegative:
                isignificand += 1 if sign else 0
            else:
                # Round to nearest
                d = fsignificand - isignificand
                if d > 0.5:
                    isignificand += 1
                elif d == 0.5:
                    # Tie
                    if rnd == RoundMode.TiesToAway:
                        isignificand += 1
                    else:
                        # All other modes tie to even
                        if _isodd(isignificand):
                            isignificand += 1

        result = isignificand * (2.0**expval)

    if result == 0:
        if sign and fi.has_nz:
            return -0.0
        else:
            return 0.0

    # Overflow
    if result > fi.max:
        if sat:
            result = fi.max
        else:
            if fi.has_infs:
                result = np.inf
            elif fi.num_nans > 0:
                result = np.nan
            else:
                raise ValueError(f"No Infs or NaNs in format {fi}, and sat=False")

    # Set sign
    if sign:
        result = -result

    return result

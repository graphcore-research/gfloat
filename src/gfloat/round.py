# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from enum import Enum
import numpy as np
import math

from .types import FormatInfo, RoundMode


def _isodd(v: int):
    return v & 0x1 == 1


def round_float(fi: FormatInfo, v: float, rnd=None) -> float:
    """
    Round input (as float, representing "infinite precision") to the given FormatInfo.

    Returns a float which exactly equals one of the code points.
    """

    # Constants
    k = fi.k
    p = fi.precision
    w = fi.expBits
    bias = fi.expBias
    t = p - 1
    rnd = rnd or fi.preferred_rounding

    assert np.isfinite(v)

    # Extract bitfield components
    sign = np.signbit(v)

    if v != 0:
        if np.isnan(v):
            if fi.num_nans == 0:
                raise ValueError(f"No NaN in format {fi}")
            return np.nan

        if np.isinf(v):
            if not fi.has_infs:
                raise ValueError(f"No Infs in format {fi}")
            return v

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
    else:
        result = 0

    if result == 0:
        if sign and fi.has_nz:
            return -0.0
        else:
            return 0.0

    # Overflow
    if rnd == RoundMode.OCP_NONSAT:
        # Check v, not result, as the spec says all values > fi.max should become inf
        if v > fi.max:
            result = np.inf if fi.has_infs else np.nan
    elif rnd == RoundMode.OCP_SAT:
        if v > fi.max:
            result = fi.max
    else:
        # Compare rounded result to fi.max, so the values between
        # fi.max and halfup(fi.max) round to fi.max
        if result > fi.max:
            if fi.has_infs:
                result = np.inf
            else:
                result = fi.max

    # Set sign
    if sign:
        result = -result

    return result

# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from enum import Enum
import numpy as np
import math

from .types import FormatInfo


def isodd(v: int):
    return v & 0x1 == 1


def iseven(v: int):
    return v & 0x1 == 0


class RoundMode(Enum):
    """
    Enum for IEEE-754 rounding modes.

    Result r is obtained from input v depending on rounding mode as follows
    """

    TowardZero = 1  #: max{r s.t. |r| <= |v|}
    TowardNegative = 2  #: max{r s.t. r <= v}
    TowardPositive = 3  #: min{r s.t. r >= v}
    TiesToEven = 4  #: See [Note]
    TiesToAway = 5  #: See [Note]


def round_float(fi: FormatInfo, v: float, rnd=RoundMode.TiesToEven) -> float:
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
                d = fsignificand - isignificand
                if d > 0.5:
                    isignificand += 1
                elif d == 0.5:
                    if (rnd == RoundMode.TiesToAway) or (
                        rnd == RoundMode.TiesToEven and isodd(isignificand)
                    ):
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
    if result > fi.max:
        if fi.has_infs:
            result = np.inf
        else:
            result = fi.max

    # Set sign
    if sign:
        result = -result

    return result

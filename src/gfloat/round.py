# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import numpy as np

from .types import FormatInfo


def round_float(fi: FormatInfo, v: float) -> float:
    """
    Round input (as float) to the given FormatInfo.

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
        biased_exp = expval + bias - 1
        effective_precision = t + min(biased_exp, 0)

        # Lift to "integer * 2^e"
        fsignificand *= 2.0**effective_precision
        expval -= effective_precision

        # round to nearest even -- use python round because numpy round is
        # "fast but sometimes inexact"
        # https://numpy.org/doc/stable/reference/generated/numpy.round.html
        fsignificand = round(fsignificand)

        result = fsignificand * (2.0**expval)
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

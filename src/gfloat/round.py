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

    if v == 0:
        if fi.has_nz and sign:
            return -0.0
        else:
            return 0.0

    if np.isnan(v):
        if fi.num_nans == 0:
            raise ValueError(f"No NaN in format {fi}")
        return np.nan

    if np.isinf(v):
        if not fi.has_infs:
            raise ValueError(f"No Infs in format {fi}")
        return v

    fsignificand, expval = np.frexp(np.abs(v))

    # fSignificand is in [0.5,1), so if normal,
    # would be multiplied by 2 and expval reduced by 1
    fsignificand *= 2
    assert fsignificand >= 1.0 and fsignificand < 2.0
    expval -= 1

    effective_precision = t + min(expval + bias - 1, 0)

    fsignificand *= 2.0**effective_precision
    expval -= effective_precision

    # round to nearest even -- use python round because numpy round is
    # "fast but sometimes inexact"
    # https://numpy.org/doc/stable/reference/generated/numpy.round.html
    fsignificand = round(fsignificand)

    val = fsignificand * (2.0**expval)

    return val

# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import numpy as np

from .types import FloatClass, FloatValue, FormatInfo


def decode_float(fi: FormatInfo, i: int) -> FloatValue:
    """
    Given :py:class:`FormatInfo` and integer code point, decode to a :py:class:`FloatValue`

    :param fi: Foating point format descriptor.
    :type fi: FormatInfo

    :param i: Integer code point, in the range :math:`0 \le i < 2^{k}`,
              where :math:`k` = ``fi.k``
    :type i: int

    :return: Decoded float value
    :rtype: FloatValue

    :raise ValueError: If i is outside the range of valid code points in fi.
    """
    k = fi.k
    p = fi.precision
    t = p - 1  # Trailing significand field width
    num_signbits = 1 if fi.is_signed else 0
    w = k - t - num_signbits  # Exponent field width

    if i < 0 or i >= 2**k:
        raise ValueError(f"Code point {i} not in range [0, 2**{k})")

    if fi.is_signed:
        signmask = 1 << (k - 1)
        signbit = 1 if i & signmask else 0
        sign = -1 if signbit else 1
    else:
        signmask = None
        signbit = None
        sign = 1

    exp = (i >> t) & ((1 << w) - 1)
    significand = i & ((1 << t) - 1)

    expBias = fi.expBias

    # t == 0 means zero mantissa bits, assume = 1 (otherwise all values are zero)
    iszero = exp == 0 and significand == 0 and t > 0
    issubnormal = fi.has_subnormals and (exp == 0) and (significand != 0)
    isnormal = not iszero and not issubnormal
    if iszero or issubnormal:
        expval = 1 - expBias
        fsignificand = significand * 2**-t
    else:
        expval = exp - expBias
        fsignificand = 1.0 + significand * 2**-t

    # val: the raw value excluding specials
    val = sign * fsignificand * 2.0**expval

    # Now overwrite the raw value with specials: Infs, NaN, -0, NaN_0
    signed_infinity = -np.inf if signbit else np.inf

    fval = val
    # All-bits-special exponent (ABSE)
    if exp == 2**w - 1:
        min_i_with_nan = 2 ** (p - 1) - fi.num_high_nans
        if significand >= min_i_with_nan:
            fval = np.nan
        if fi.has_infs and significand == min_i_with_nan - 1:
            fval = signed_infinity

    # Negative zero or NaN
    if iszero and i == signmask:
        if fi.has_nz:
            fval = -0.0
        else:
            fval = np.nan

    # Compute FloatClass
    fclass = None
    if fval == 0:
        fclass = FloatClass.ZERO
    elif np.isnan(fval):
        fclass = FloatClass.NAN
    elif np.isfinite(fval):
        if isnormal:
            fclass = FloatClass.NORMAL
        else:
            fclass = FloatClass.SUBNORMAL
    else:
        fclass = FloatClass.INFINITE

    return FloatValue(i, fval, exp, expval, significand, fsignificand, signbit, fclass)

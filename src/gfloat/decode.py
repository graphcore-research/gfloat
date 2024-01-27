from .types import FormatInfo, FloatValue, FloatClass

import numpy as np


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
    t = p - 1  # trailing significand field width
    w = k - p

    if i < 0 or i >= 2**k:
        raise ValueError(f"Code point {i} not in range [0, 2**{k})")

    signmask = 1 << (k - 1)
    signbit = 1 if i & signmask else 0
    sign = -1 if signbit else 1

    exp = (i & (signmask - 1)) >> t
    significand = i & ((1 << t) - 1)

    expBias = fi.expBias

    isnormal = exp != 0
    if isnormal:
        expval = exp - expBias
        fsignificand = 1.0 + significand * 2**-t
    else:
        expval = 1 - expBias
        fsignificand = significand * 2**-t

    # val: the raw value excluding specials
    val = sign * fsignificand * 2**expval

    # valstr: string representation of value in base 10
    # If the representation does not roundtrip to the value,
    # it is preceded by a "~" to indicate "approximately equal to"
    valstr = f"{val}"
    if len(valstr) > 14:
        valstr = f"{val:.8}"
    if float(valstr) != val:
        valstr = "~" + valstr

    # Now overwrite the raw value with specials: Infs, NaN, -0, NaN_0
    signed_infinity = -np.inf if signbit else np.inf

    fval = val
    # All-bits-one exponent (ABOE)
    if exp == 2**w - 1:
        min_i_with_nan = 2 ** (p - 1) - fi.num_high_nans
        if significand >= min_i_with_nan:
            fval = np.nan
        if fi.has_infs and significand == min_i_with_nan - 1:
            fval = signed_infinity

    # Negative zero or NaN
    if i == signmask:
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

    # update valstr if a special value
    if fclass not in (FloatClass.ZERO, FloatClass.SUBNORMAL, FloatClass.NORMAL):
        valstr = str(fval)

    return FloatValue(
        i,
        fval,
        valstr,
        exp,
        expval,
        significand,
        fsignificand,
        signbit,
        fclass,
        fi,
    )

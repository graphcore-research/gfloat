import numpy as np


# pythran export decode_core((int, int, int, bool, bool, int, bool), int)
def decode_core(fi, i):
    k, p, expBias, has_subnormals, has_infs, num_high_nans, has_nz = fi
    t = p - 1  # trailing significand field width
    w = k - p

    if i < 0 or i >= 2**k:
        raise ValueError(f"Code point {i:d} not in range [0, 2**{k:d})")

    signmask = 1 << (k - 1)
    signbit = 1 if i & signmask else 0
    sign = -1 if signbit else 1

    exp = (i & (signmask - 1)) >> t
    significand = i & ((1 << t) - 1)

    iszero = exp == 0 and significand == 0
    issubnormal = has_subnormals and (exp == 0) and (significand != 0)
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
        min_i_with_nan = 2 ** (p - 1) - num_high_nans
        if significand >= min_i_with_nan:
            fval = np.nan
        if has_infs and significand == min_i_with_nan - 1:
            fval = signed_infinity

    # Negative zero or NaN
    if i == signmask:
        if has_nz:
            fval = -0.0
        else:
            fval = np.nan

    return fval, isnormal, exp, expval, significand, fsignificand, signbit

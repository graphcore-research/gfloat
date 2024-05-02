# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import fractions

import numpy as np


def float_pow2str(v: float, min_exponent: float = -np.inf) -> str:
    """
    Render floating point values as exact fractions times a power of two.

    Example: float_pow2str(127.0) is "127/64*2^6",

    That is (a significand between 1 and 2) times (a power of two).

    If `min_exponent` is supplied, then values with exponent below `min_exponent`,
    are printed as fractions less than 1, with exponent set to `min_exponent`.
    This is typically used to represent subnormal values.

    """
    if not np.isfinite(v):
        return str(v)

    s = np.sign(v)
    x = np.abs(v)
    e = np.floor(np.log2(x))
    sig = x * 2.0**-e
    if e < min_exponent:
        sig *= 2.0 ** (e - min_exponent)
        e = min_exponent

    significand = fractions.Fraction(sig)
    return ("-" if s < 0 else "") + f"{significand}*2^{int(e):d}"


def float_tilde_unless_roundtrip_str(v: float, width: int = 14, d: int = 8) -> str:
    """
    Return a string representation of :paramref:`v`, in base 10,
    with maximum width :paramref:`width` and decimal digits :paramref:`d`


    """
    # valstr: string representation of value in base 10
    # If the representation does not roundtrip to the value,
    # it is preceded by a "~" to indicate "approximately equal to"
    s = f"{v}"
    if len(s) > width:
        if abs(v) < 1 and not "e" in s:
            s = f"{v:.{d}f}"
        else:
            s = f"{v:.{d}}"
    if np.isfinite(v) and float(s) != v:
        s = "~" + s

    return s

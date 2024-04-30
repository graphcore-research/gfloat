import fractions
import numpy as np


def float_pow2str(v, min_exponent=-np.inf):
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
    e = int(np.floor(np.log2(x)))
    sig = x * 2**-e
    if e < min_exponent:
        sig *= 2 ** (e - min_exponent)
        e = min_exponent

    significand = fractions.Fraction(sig)
    return ("-" if s < 0 else "") + f"{significand}*2^{e:d}"

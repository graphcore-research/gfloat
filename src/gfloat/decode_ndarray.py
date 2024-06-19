# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import numpy as np

from .types import FloatClass, FloatValue, FormatInfo


def decode_ndarray(fi: FormatInfo, codes: np.ndarray, np=np) -> np.ndarray:
    r"""
    Vectorized version of :function:`decode_float`

    Args:
      fi (FormatInfo): Floating point format descriptor.
      i (array of int):  Integer code points, in the range :math:`0 \le i < 2^{k}`,
                where :math:`k` = ``fi.k``

    Returns:
      Decoded float values

    Raises:
      ValueError:
        If any :paramref:`i` is outside the range of valid code points in :paramref:`fi`.
    """
    assert np.issubdtype(codes.dtype, np.integer)

    k = fi.k
    p = fi.precision
    t = p - 1  # Trailing significand field width
    num_signbits = 1 if fi.is_signed else 0
    w = k - t - num_signbits  # Exponent field width

    if np.any(codes < 0) or np.any(codes >= 2**k):
        raise ValueError(f"Code point not in range [0, 2**{k})")

    if fi.is_signed:
        signmask = 1 << (k - 1)
        sign = np.where(codes & signmask, -1.0, 1.0)
    else:
        signmask = None
        sign = 1.0

    exp = (codes >> t) & ((1 << w) - 1)
    significand = codes & ((1 << t) - 1)
    if fi.is_twos_complement:
        significand = np.where(sign < 0, (1 << t) - significand, significand)

    expBias = fi.expBias

    iszero = (exp == 0) & (significand == 0) if fi.has_zero else False
    issubnormal = (exp == 0) & (significand != 0) if fi.has_subnormals else False
    isnormal = ~iszero & ~issubnormal
    expval = np.where(~isnormal, 1 - expBias, exp - expBias)
    fsignificand = np.where(~isnormal, significand * 2**-t, 1.0 + significand * 2**-t)

    # Normal/Subnormal/Zero case, other values will be overwritten
    fval = np.where(iszero, 0.0, sign * fsignificand * 2.0**expval)

    # All-bits-special exponent (ABSE)
    if w > 0:
        abse = exp == 2**w - 1
        min_i_with_nan = 2 ** (p - 1) - fi.num_high_nans
        fval = np.where(abse & (significand >= min_i_with_nan), np.nan, fval)
        if fi.has_infs:
            fval = np.where(
                abse & (significand == min_i_with_nan - 1), np.inf * sign, fval
            )

    # Negative zero
    if fi.has_nz:
        fval = np.where(iszero & (sign < 0), -0.0, fval)
    else:
        # Negative zero slot is nan
        fval = np.where(codes == fi.code_of_negzero, np.nan, fval)

    return fval

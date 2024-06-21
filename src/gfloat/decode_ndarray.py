# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from types import ModuleType
import numpy as np
from .types import FormatInfo


def decode_ndarray(
    fi: FormatInfo, codes: np.ndarray, np: ModuleType = np
) -> np.ndarray:
    r"""
    Vectorized version of :meth:`decode_float`

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

    exp = ((codes >> t) & ((1 << w) - 1)).astype(np.int64)
    significand = codes & ((1 << t) - 1)
    if fi.is_twos_complement:
        significand = np.where(sign < 0, (1 << t) - significand, significand)

    expBias = fi.expBias

    iszero = (exp == 0) & (significand == 0) & fi.has_zero
    issubnormal = (exp == 0) & (significand != 0) & fi.has_subnormals
    isnormal = ~iszero & ~issubnormal
    expval = np.where(~isnormal, 1 - expBias, exp - expBias)
    fsignificand = np.where(~isnormal, significand * 2**-t, 1.0 + significand * 2**-t)

    # Normal/Subnormal/Zero case, other values will be overwritten
    fval = np.where(iszero, 0.0, sign * fsignificand * 2.0**expval)

    if fi.has_infs:
        fval = np.where(codes == fi.code_of_posinf, np.inf, fval)
        fval = np.where(codes == fi.code_of_neginf, -np.inf, fval)

    if fi.num_nans > 0:
        code_is_nan = codes == fi.code_of_nan
        if w > 0:
            # All-bits-special exponent (ABSE)
            abse = exp == 2**w - 1
            min_code_with_nan = 2 ** (p - 1) - fi.num_high_nans
            code_is_nan |= abse & (significand >= min_code_with_nan)

        fval = np.where(code_is_nan, np.nan, fval)

    # Negative zero
    if fi.has_nz:
        fval = np.where(iszero & (sign < 0), -0.0, fval)

    return fval

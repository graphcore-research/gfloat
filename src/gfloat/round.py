# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import math

import numpy as np

from .types import FormatInfo, RoundMode


def _isodd(v: int) -> bool:
    return v & 0x1 == 1


def round_float(
    fi: FormatInfo, v: float, rnd: RoundMode = RoundMode.TiesToEven, sat: bool = False
) -> float:
    """
    Round input to the given :py:class:`FormatInfo`, given rounding mode and saturation flag

    An input NaN will convert to a NaN in the target.
    An input Infinity will convert to the largest float if :paramref:`sat`,
    otherwise to an Inf, if present, otherwise to a NaN.
    Negative zero will be returned if the format has negative zero, otherwise zero.

    Args:
      fi (FormatInfo): Describes the target format
      v (float): Input value to be rounded
      rnd (RoundMode): Rounding mode to use
      sat (bool): Saturation flag: if True, round overflowed values to `fi.max`

    Returns:
      A float which is one of the values in the format.

    Raises:
       ValueError: The target format cannot represent the input
             (e.g. converting a `NaN`, or an `Inf` when the target has no
             `NaN` or `Inf`, and :paramref:`sat` is false)
    """

    # Constants
    p = fi.precision
    bias = fi.expBias

    if np.isnan(v):
        if fi.num_nans == 0:
            raise ValueError(f"No NaN in format {fi}")

        # Note that this does not preserve the NaN payload
        return np.nan

    # Extract sign
    sign = np.signbit(v) and fi.is_signed
    vpos = -v if sign else v

    if np.isinf(vpos):
        result = np.inf

    elif vpos == 0:
        result = 0

    else:
        # Extract exponent
        expval = int(np.floor(np.log2(vpos)))

        # Effective precision, accounting for right shift for subnormal values
        if fi.has_subnormals:
            expval = max(expval, 1 - bias)

        # Lift to "integer * 2^e"
        expval = expval - p + 1

        # use ldexp instead of vpos*2**-expval to avoid overflow
        fsignificand = math.ldexp(vpos, -expval)

        # Round
        isignificand = math.floor(fsignificand)
        delta = fsignificand - isignificand
        if (
            (rnd == RoundMode.TowardPositive and not sign and delta > 0)
            or (rnd == RoundMode.TowardNegative and sign and delta > 0)
            or (rnd == RoundMode.TiesToAway and delta >= 0.5)
            or (rnd == RoundMode.TiesToEven and delta > 0.5)
            or (rnd == RoundMode.TiesToEven and delta == 0.5 and _isodd(isignificand))
        ):
            isignificand += 1

        ## Special case for Precision=1, all-log format with zero.
        #  The logic is simply duplicated (and isignificand overwritten) for clarity.
        if fi.precision == 1:
            isignificand = math.floor(fsignificand)
            code_is_odd = isignificand != 0 and _isodd(expval + bias)
            if (
                (rnd == RoundMode.TowardPositive and not sign and delta > 0)
                or (rnd == RoundMode.TowardNegative and sign and delta > 0)
                or (rnd == RoundMode.TiesToAway and delta >= 0.5)
                or (rnd == RoundMode.TiesToEven and delta > 0.5)
                or (rnd == RoundMode.TiesToEven and delta == 0.5 and code_is_odd)
            ):
                # Go to nextUp.
                # Increment isignificand if zero,
                # else increment exponent
                if isignificand == 0:
                    isignificand = 1
                else:
                    assert isignificand == 1
                    expval += 1
            ## End special case for Precision=1.

        # Reconstruct rounded result to float
        result = isignificand * (2.0**expval)

    if result == 0:
        if sign and fi.has_nz:
            return -0.0
        else:
            return 0.0

    # Overflow
    amax = -fi.min if sign else fi.max
    if result > amax:
        if (
            sat
            or (rnd == RoundMode.TowardNegative and not sign and np.isfinite(v))
            or (rnd == RoundMode.TowardPositive and sign and np.isfinite(v))
            or (rnd == RoundMode.TowardZero and np.isfinite(v))
        ):
            result = amax
        else:
            if fi.has_infs:
                result = np.inf
            elif fi.num_nans > 0:
                result = np.nan
            else:
                raise ValueError(f"No Infs or NaNs in format {fi}, and sat=False")

    # Set sign
    if sign:
        result = -result

    return result


def encode_float(fi: FormatInfo, v: float) -> int:
    """
    Encode input to the given :py:class:`FormatInfo`.

    Will round toward zero if :paramref:`v` is not in the value set.
    Will saturate to `Inf`, `NaN`, `fi.max` in order of precedence.
    Encode -0 to 0 if not `fi.has_nz`

    For other roundings and saturations, call :func:`round_float` first.

    Args:
      fi (FormatInfo): Describes the target format
      v (float): The value to be encoded.

    Returns:
      The integer code point
    """

    # Format Constants
    k = fi.bits
    p = fi.precision
    t = p - 1

    # Encode
    if np.isnan(v):
        return fi.code_of_nan

    # Overflow/underflow
    if v > fi.max:
        if fi.has_infs:
            return fi.code_of_posinf
        if fi.num_nans > 0:
            return fi.code_of_nan
        return fi.code_of_max

    if v < fi.min:
        if fi.has_infs:
            return fi.code_of_neginf
        if fi.num_nans > 0:
            return fi.code_of_nan
        return fi.code_of_min

    # Finite values
    sign = fi.is_signed and np.signbit(v)
    vpos = -v if sign else v

    if fi.has_subnormals and vpos <= fi.smallest_subnormal / 2:
        isig = 0
        biased_exp = 0
    else:
        sig, exp = np.frexp(vpos)
        exp = int(exp)  # All calculations in Python ints

        # sig in range [0.5, 1)
        sig *= 2
        exp -= 1
        # now sig in range [1, 2)

        biased_exp = exp + fi.expBias
        if biased_exp < 1 and fi.has_subnormals:
            # subnormal
            sig *= 2.0 ** (biased_exp - 1)
            biased_exp = 0
            assert vpos == sig * 2 ** (1 - fi.expBias)
        else:
            if sig > 0:
                sig -= 1.0

        isig = math.floor(sig * 2**t)

    # Zero
    if isig == 0 and biased_exp == 0 and fi.has_zero:
        if sign and fi.has_nz:
            return fi.code_of_negzero
        else:
            return fi.code_of_zero

    # Nonzero
    assert isig < 2**t
    assert biased_exp < 2**fi.expBits or fi.is_twos_complement

    # Handle two's complement encoding
    if fi.is_twos_complement and sign:
        isig = (1 << t) - isig

    # Pack values into a single integer
    code = (int(sign) << (k - 1)) | (biased_exp << t) | (isig << 0)

    return code

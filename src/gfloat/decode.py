# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from .types import FormatInfo, FloatValue, FloatClass

import numpy as np

from .decode_core import decode_core


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

    fi_tuple = (
        fi.k,
        fi.precision,
        fi.expBias,
        fi.has_subnormals,
        fi.has_infs,
        fi.num_high_nans,
        fi.has_nz,
    )
    fval, isnormal, exp, expval, significand, fsignificand, signbit = decode_core(
        fi_tuple, i
    )
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

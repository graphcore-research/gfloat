# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import ml_dtypes
import numpy as np
import pytest

from gfloat import decode_float, encode_float
from gfloat.formats import *


@pytest.mark.parametrize("fi", all_formats, ids=str)
def test_encode(fi):
    dec = lambda v: decode_float(fi, v).fval

    if fi.bits <= 8:
        step = 1
    elif fi.bits <= 16:
        step = 13
    elif fi.bits <= 32:
        step = 73013

    for i in range(0, 2**fi.bits, step):
        fv = decode_float(fi, i)
        ival = encode_float(fi, fv.fval)
        fv2 = decode_float(fi, ival)
        assert (i == ival) or np.isnan(fv.fval)
        np.testing.assert_equal(fv2.fval, fv.fval)

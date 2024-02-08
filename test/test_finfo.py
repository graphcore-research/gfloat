# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

# Test that finfo methods on FloatFormat agree with numpy/ml_dtypes

import pytest
import ml_dtypes
import numpy as np

from gfloat import decode_float, round_float
from gfloat.formats import *


@pytest.mark.parametrize(
    "fmt,npfmt",
    [
        (format_info_ocp_e5m2, ml_dtypes.float8_e5m2),
        (format_info_ocp_e4m3, ml_dtypes.float8_e4m3fn),
        (format_info_binary16, np.float16),
        (format_info_bfloat16, ml_dtypes.bfloat16),
    ],
)
def test_finfo(fmt, npfmt):
    assert fmt.eps == ml_dtypes.finfo(npfmt).eps
    assert fmt.epsneg == ml_dtypes.finfo(npfmt).epsneg
    assert fmt.max == ml_dtypes.finfo(npfmt).max
    assert fmt.maxexp == ml_dtypes.finfo(npfmt).maxexp

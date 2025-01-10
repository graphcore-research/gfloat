# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import array_api_strict as xp
import numpy as np
import pytest

from gfloat import RoundMode, decode_float, decode_ndarray, round_float, round_ndarray
from gfloat.formats import *

xp.set_array_api_strict_flags(api_version="2024.12")

# Hack until https://github.com/data-apis/array-api/issues/807
_xp_where = xp.where
xp.where = lambda a, t, f: _xp_where(a, xp.asarray(t), xp.asarray(f))
_xp_maximum = xp.maximum
xp.maximum = lambda a, b: _xp_maximum(xp.asarray(a), xp.asarray(b))


@pytest.mark.parametrize("fi", all_formats)
@pytest.mark.parametrize("rnd", RoundMode)
@pytest.mark.parametrize("sat", [True, False])
def test_array_api(fi, rnd, sat):
    a = np.random.rand(23, 1, 34) - 0.5
    a = xp.asarray(a)

    srnumbits = 32
    srbits = np.random.randint(0, 2**srnumbits, a.shape)
    srbits = xp.asarray(srbits)

    round_ndarray(fi, a, rnd, sat, srbits=srbits, srnumbits=srnumbits)

# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest

from gfloat import decode_block, encode_block
from gfloat.formats import *


@pytest.mark.parametrize("fi", all_block_formats, ids=str)
def test_blocks(fi: BlockFormatInfo) -> None:

    vals = np.linspace(-37.0, 42.0, 32)

    scale = 8.0
    block = list(encode_block(fi, scale, vals))
    decoded_vals = list(decode_block(fi, block))

    atol = 2 * scale * fi.etype.eps
    np.testing.assert_allclose(decoded_vals, vals, atol=atol)

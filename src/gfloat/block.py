# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

# Block floating point formats
# https://en.wikipedia.org/wiki/Block_floating_point

from dataclasses import dataclass
from typing import Iterable, Iterator

from .decode import decode_float
from .round import encode_float, round_float
from .types import FloatValue, FormatInfo


@dataclass
class BlockFormatInfo:

    #: Short name for the format, e.g. BlockFP8
    name: str

    #: Element data type
    etype: FormatInfo

    #: Scaling block size
    k: int

    #: Scale datatype
    stype: FormatInfo

    #: ## Derived values

    @property
    def element_bits(self) -> int:
        """The number of bits in each element, d"""
        return self.etype.k

    @property
    def scale_bits(self) -> int:
        """The number of bits in the scale, w"""
        return self.stype.k

    @property
    def block_size_bytes(self) -> int:
        """The number of bytes in a block"""
        bits = self.element_bits * self.k + self.scale_bits
        assert bits % 8 == 0
        return bits // 8

    def __str__(self):
        return f"{self.name}"


def decode_block(fi: BlockFormatInfo, block: Iterable[int]) -> Iterable[float]:
    """
    Decode a `param:block` of integer codepoints in Block Format `param:fi`

    The scale is assumed to be at the front of the block
    """
    it = iter(block)

    scale_encoding = next(it)
    scale = decode_float(fi.stype, scale_encoding).fval

    for val_encoding in it:
        val = scale * decode_float(fi.etype, val_encoding).fval
        yield val

    # TODO: Assert length of block was k+1?  Messy unless block is len()able


def encode_block(
    fi: BlockFormatInfo, scale: float, vals: Iterable[float]
) -> Iterable[int]:
    """
    Encode a `param:block` of bytes into Block Format `param:fi`

    The desired scale is explicitly passed, and is converted to 1/(1/scale)
    before rounding to the target format.
    It is checked for overflow in the target format,
    and will raise an exception if it does.
    """
    recip_scale = 1 / scale
    scale = 1 / recip_scale

    if scale > fi.stype.max:
        raise ValueError(f"Scaled {scale} too large for {fi.stype}")

    enc = lambda ty, x: encode_float(ty, round_float(ty, x))

    yield enc(fi.stype, scale)

    for val in vals:
        yield enc(fi.etype, recip_scale * val)

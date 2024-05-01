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
    Decode a :paramref:`block` of integer codepoints in Block Format :paramref:`fi`

    The scale is encoded in the first value of :paramref:`block`,
    with the remaining values encoding the block elements.

    The size of the iterable is not checked against the format descriptor.

    :param fi: Describes the block format
    :type fi: BlockFormatInfo

    :param block: Input block
    :type block: Iterable[int]

    :return: A sequence of floats representing the encoded values.
    :rtype: Iterable[float]
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
    Encode a :paramref:`block` of bytes into block Format descibed by :paramref:`fi`

    The :paramref:`scale` is explicitly passed, and is converted to `1/(1/scale)`
    before rounding to the target format.

    It is checked for overflow in the target format,
    and will raise an exception if it does.

    :param fi: Describes the target block format
    :type fi: BlockFormatInfo

    :param scale: Scale to be recorded in the block
    :type scale: float

    :param vals: Input block
    :type vals: Iterable[int]

    :return: A sequence of ints representing the encoded values.
    :rtype: Iterable[int]

    :raises ValueError: The scale overflows the target scale encoding format.

    """
    recip_scale = 1 / scale
    scale = 1 / recip_scale

    if scale > fi.stype.max:
        raise ValueError(f"Scaled {scale} too large for {fi.stype}")

    enc = lambda ty, x: encode_float(ty, round_float(ty, x))

    yield enc(fi.stype, scale)

    for val in vals:
        yield enc(fi.etype, recip_scale * val)

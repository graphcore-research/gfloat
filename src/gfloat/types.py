# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from dataclasses import dataclass
from enum import Enum
import numpy as np


@dataclass
class FormatInfo:
    """
    Class describing a floating-point format, parametrized
    by width, precision, and special value encoding rules.

    """

    #: Short name for the format, e.g. binary32, bfloat16
    name: str

    #: Number of bits in the format
    k: int

    #: Number of significand bits (including implicit leading bit)
    precision: int

    #: Largest exponent, emax, which shall equal floor(log_2(maxFinite))
    emax: int

    #: Set if format encodes -0 at (sgn=1,exp=0,significand=0).
    #: If False, that encoding decodes to a NaN labelled NaN_0
    has_nz: bool

    #: Set if format includes +/- Infinity.
    #: If set, the non-nan value with the highest encoding for each sign (s)
    #: is replaced by (s)Inf.
    has_infs: bool

    #: Number of NaNs that are encoded in the highest encodings for each sign
    num_high_nans: int

    #: Set if format encodes subnormals
    has_subnormals: bool

    #: ## Derived values

    @property
    def tSignificandBits(self):
        """The number of trailing significand bits, t"""
        return self.precision - 1

    @property
    def expBits(self):
        """The number of exponent bits, w"""
        return self.k - self.precision

    @property
    def expBias(self):
        """The exponent bias derived from (p,emax)

        This is the bias that should be applied so that
           :math:`floor(log_2(maxFinite)) = emax`
        """
        # Calculate whether all of the all-bits-one-exponent values contain specials.
        # If so, emax will be obtained for exponent value 2^w-2, otherwise it is 2^w-1
        t = self.tSignificandBits
        num_posinfs = 1 if self.has_infs else 0
        all_bits_one_full = (self.num_high_nans + num_posinfs == 2**t) or (
            self.expBits == 0 and self.has_infs
        )

        # Compute exponent bias.
        exp_for_emax = 2**self.expBits - (2 if all_bits_one_full else 1)
        return exp_for_emax - self.emax

    @property
    def num_nans(self):
        """The number of code points which decode to NaN"""
        return (0 if self.has_nz else 1) + 2 * self.num_high_nans


class FloatClass(Enum):
    """
    Enum for the classification of a FloatValue.
    """

    NORMAL = 1  #: A positive or negative normalized non-zero value
    SUBNORMAL = 2  #: A positive or negative subnormal value
    ZERO = 3  #: A positive or negative zero value
    INFINITE = 4  #: A positive or negative infinity (+/-Inf)
    NAN = 5  #: Not a Number (NaN)


@dataclass
class FloatValue:
    """
    A floating-point value decoded in great detail.
    """

    ival: int  #: Integer code point

    #: Value. Assumed to be exactly round-trippable to python float.
    #: This is true for all <64bit formats known in 2023.
    fval: float

    val_raw: float  #: Value, assuming all code points finite
    exp: int  #: Raw exponent without bias
    expval: int  #: Exponent, bias subtracted
    significand: int  #: Significand as an integer
    fsignificand: float  #: Significand as a float in the range [0,2)
    signbit: int  #: Sign bit: 1 => negative, 0 => positive
    fclass: FloatClass  #: See FloatClass
    fi: FormatInfo  # Backlink to FormatInfo

    @property
    def signstr(self):
        """Return "+" or "-" according to signbit"""
        return "-" if self.signbit else "+"

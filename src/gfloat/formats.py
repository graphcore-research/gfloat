# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from .block import BlockFormatInfo
from .types import FormatInfo, Domain

import math

#: FormatInfo for IEEE-754 Binary64 format
format_info_binary64 = FormatInfo(
    name="binary64",
    k=64,
    precision=53,
    bias=2 ** (64 - 53 - 1) - 1,
    has_nz=True,
    domain=Domain.Extended,
    num_high_nans=2**52 - 1,
    has_subnormals=True,
    is_signed=True,
    is_twos_complement=False,
)

#: FormatInfo for IEEE-754 Binary32 format
format_info_binary32 = FormatInfo(
    name="binary32",
    k=32,
    precision=24,
    bias=2 ** (32 - 24 - 1) - 1,
    has_nz=True,
    domain=Domain.Extended,
    num_high_nans=2**23 - 1,
    has_subnormals=True,
    is_signed=True,
    is_twos_complement=False,
)

#: FormatInfo for IEEE-754 Binary16 format
format_info_binary16 = FormatInfo(
    name="binary16",
    k=16,
    precision=11,
    bias=2 ** (16 - 11 - 1) - 1,
    has_nz=True,
    domain=Domain.Extended,
    num_high_nans=2**10 - 1,
    has_subnormals=True,
    is_signed=True,
    is_twos_complement=False,
)

#: FormatInfo for Google BFloat16 format
format_info_bfloat16 = FormatInfo(
    name="bfloat16",
    k=16,
    precision=8,
    bias=2 ** (16 - 8 - 1) - 1,
    has_nz=True,
    domain=Domain.Extended,
    num_high_nans=2**7 - 1,
    has_subnormals=True,
    is_signed=True,
    is_twos_complement=False,
)

#: FormatInfo for OCP E5M2 format
format_info_ocp_e5m2 = FormatInfo(
    name="ocp_e5m2",
    k=8,
    precision=3,
    bias=2 ** (8 - 3 - 1) - 1,
    has_nz=True,
    domain=Domain.Extended,
    num_high_nans=2**2 - 1,
    has_subnormals=True,
    is_signed=True,
    is_twos_complement=False,
)

#: FormatInfo for OCP E4M3 format
format_info_ocp_e4m3 = FormatInfo(
    name="ocp_e4m3",
    k=8,
    precision=4,
    bias=2 ** (8 - 4 - 1) - 1,
    has_nz=True,
    domain=Domain.Finite,
    num_high_nans=1,
    has_subnormals=True,
    is_signed=True,
    is_twos_complement=False,
)

#: FormatInfo for OCP MX E2M3 format
format_info_ocp_e2m3 = FormatInfo(
    name="ocp_e2m3",
    k=6,
    precision=4,
    bias=2 ** (6 - 4 - 1) - 1,
    has_nz=True,
    domain=Domain.Finite,
    num_high_nans=0,
    has_subnormals=True,
    is_signed=True,
    is_twos_complement=False,
)

#: FormatInfo for OCP MX E3M2 format
format_info_ocp_e3m2 = FormatInfo(
    name="ocp_e3m2",
    k=6,
    precision=3,
    bias=2 ** (6 - 3 - 1) - 1,
    has_nz=True,
    domain=Domain.Finite,
    num_high_nans=0,
    has_subnormals=True,
    is_signed=True,
    is_twos_complement=False,
)

#: FormatInfo for OCP MX E2M1 format
format_info_ocp_e2m1 = FormatInfo(
    name="ocp_e2m1",
    k=4,
    precision=2,
    bias=2 ** (4 - 2 - 1) - 1,
    has_nz=True,
    domain=Domain.Finite,
    num_high_nans=0,
    has_subnormals=True,
    is_signed=True,
    is_twos_complement=False,
)

#: FormatInfo for OCP MX E8M0 format
format_info_ocp_e8m0 = FormatInfo(
    name="ocp_e8m0",
    k=8,
    precision=1,
    bias=2 ** (8 - 1) - 1,
    has_nz=False,
    domain=Domain.Finite,
    num_high_nans=1,
    has_subnormals=False,
    is_signed=False,
    is_twos_complement=False,
)

#: FormatInfo for OCP MX INT8 format
format_info_ocp_int8 = FormatInfo(
    name="ocp_int8",
    k=8,
    precision=8,
    bias=0,
    has_nz=False,
    domain=Domain.Finite,
    num_high_nans=0,
    has_subnormals=True,
    is_signed=True,
    is_twos_complement=True,
)


def format_info_p3109(
    k: int,
    precision: int,
    domain: Domain = Domain.Extended,
    signedness: bool = True,
    may25bias: bool = False,
) -> FormatInfo:
    """
    FormatInfo for P3109 K{k} P{p} formats

    Args:
      k (int): Format width in bits
      p (int): Precision in bits
      domain (Domain): Extended (default) or finite
      signedness (bool): True (default) if signed, False if unsigned
      may25bias (bool): False (default) for may25 bias

    Returns:
       FormatInfo class describing the format

    Raises:
       ValueError: If p is not in 1..k
       ValueError: If k is < 2
    """
    if precision < 1 or precision > k:
        raise ValueError(f"P3109 format not defined for k={k}, p={precision}")

    if k < 2:
        raise ValueError(f"P3109 format not defined for k={k} < 2")

    estr = "e" if domain == Domain.Extended else "f"
    sstr = "s" if signedness else "u"
    v = "" if not may25bias else "mtfb_"
    name = f"{v}p3109_k{k}p{precision}{estr}{sstr}"
    if may25bias:
        bias = math.floor(2 ** (k - precision - 1) - 1)
    else:
        if signedness:
            bias = math.floor(2 ** (k - precision - 1))
        else:
            bias = 2 ** (k - precision)

    return FormatInfo(
        name,
        k=k,
        precision=precision,
        bias=bias,
        has_nz=False,
        domain=domain,
        num_high_nans=0 if signedness else 1,
        has_subnormals=True,
        is_signed=signedness,
        is_twos_complement=False,
    )


# Collections of formats
_tiny_formats = [
    format_info_ocp_e2m1,
    format_info_p3109(4, 2, Domain.Finite),
    format_info_ocp_e2m3,
    format_info_ocp_e3m2,
    format_info_p3109(6, 3, Domain.Finite),
    format_info_p3109(6, 4, Domain.Finite),
]

p3109_binary8_formats = [
    format_info_p3109(8, p, domain, signedness)
    for p in (1, 3, 4)
    for signedness in (True, False)
    for domain in (Domain.Extended, Domain.Finite)
]

_fp8_formats = [
    format_info_ocp_e4m3,
    format_info_ocp_e5m2,
    *p3109_binary8_formats,
]

_fp16_formats = [
    format_info_binary16,
    format_info_bfloat16,
]

all_formats = [
    *_tiny_formats,
    *_fp8_formats,
    *_fp16_formats,
    format_info_binary32,
    format_info_binary64,
    format_info_ocp_e8m0,
    format_info_ocp_int8,
]

# ------
# Block formats

format_info_mxfp8_e5m2 = BlockFormatInfo(
    "mxfp8_e5m2", format_info_ocp_e5m2, 32, format_info_ocp_e8m0
)

format_info_mxfp8_e4m3 = BlockFormatInfo(
    "mxfp8_e4m3", format_info_ocp_e4m3, 32, format_info_ocp_e8m0
)

format_info_mxfp6_e3m2 = BlockFormatInfo(
    "mxfp6_e3m2", format_info_ocp_e3m2, 32, format_info_ocp_e8m0
)

format_info_mxfp6_e2m3 = BlockFormatInfo(
    "mxfp6_e2m3", format_info_ocp_e2m3, 32, format_info_ocp_e8m0
)

format_info_mxfp4_e2m1 = BlockFormatInfo(
    "mxfp4_e2m1", format_info_ocp_e2m1, 32, format_info_ocp_e8m0
)

format_info_mxfp4_e2m1 = BlockFormatInfo(
    "mxfp4_e2m1", format_info_ocp_e2m1, 32, format_info_ocp_e8m0
)

format_info_mxint8 = BlockFormatInfo(
    "mxint8", format_info_ocp_int8, 32, format_info_ocp_e8m0
)

all_block_formats = [
    format_info_mxfp8_e5m2,
    format_info_mxfp8_e4m3,
    format_info_mxfp6_e3m2,
    format_info_mxfp6_e2m3,
    format_info_mxfp4_e2m1,
    format_info_mxint8,
]

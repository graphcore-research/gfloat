# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from .block import BlockFormatInfo
from .types import FormatInfo

#: FormatInfo for IEEE-754 Binary64 format
format_info_binary64 = FormatInfo(
    name="binary64",
    k=64,
    precision=53,
    emax=1023,
    has_nz=True,
    has_infs=True,
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
    emax=127,
    has_nz=True,
    has_infs=True,
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
    emax=15,
    has_nz=True,
    has_infs=True,
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
    emax=127,
    has_nz=True,
    has_infs=True,
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
    emax=15,
    has_nz=True,
    has_infs=True,
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
    emax=8,
    has_nz=True,
    has_infs=False,
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
    emax=2,
    has_nz=True,
    has_infs=False,
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
    emax=4,
    has_nz=True,
    has_infs=False,
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
    emax=2,
    has_nz=True,
    has_infs=False,
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
    emax=127,
    has_nz=False,
    has_infs=False,
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
    emax=0,
    has_nz=False,
    has_infs=False,
    num_high_nans=0,
    has_subnormals=True,
    is_signed=True,
    is_twos_complement=True,
)


def format_info_p3109(precision: int) -> FormatInfo:
    """
    FormatInfo for P3109 P{p} formats

    Args:
      p (int): Precision in bits

    Returns:
       FormatInfo class describing the format

    Raises:
       ValueError: If p is not in 1..7
    """
    if precision < 1 or precision > 7:
        raise ValueError(f"P3109 format not defined for p={precision}")

    name = f"p3109_p{precision}"
    emax = 2 ** (7 - precision) - 1

    return FormatInfo(
        name,
        k=8,
        precision=precision,
        emax=emax,
        has_nz=False,
        has_infs=True,
        num_high_nans=0,
        has_subnormals=True,
        is_signed=True,
        is_twos_complement=False,
    )


# Collections of formats
_tiny_formats = [
    format_info_ocp_e2m1,
    format_info_ocp_e2m3,
    format_info_ocp_e3m2,
]

p3109_formats = [format_info_p3109(p) for p in range(1, 7)]

_fp8_formats = [
    format_info_ocp_e4m3,
    format_info_ocp_e5m2,
    *p3109_formats,
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

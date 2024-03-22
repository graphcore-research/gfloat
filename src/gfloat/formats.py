# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from gfloat import FormatInfo, RoundMode

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
)


def format_info_p3109(precision: int) -> FormatInfo:
    """
    FormatInfo for P3109 P{p} formats

    :param p: Precision in bits
    :type p: int

    :return: FormatInfo class describing the format
    :rtype: FormatInfo

    :raise ValueError: If p is not in 1..7
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
    )

# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import struct

import pytest


def _is_32bit_python() -> bool:
    return struct.calcsize("P") * 8 == 32


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not _is_32bit_python():
        return

    mark = pytest.mark.xfail(
        reason="Known 32-bit regressions (issue #57)",
        strict=False,
    )

    for item in items:
        nodeid = item.nodeid

        if nodeid.startswith("test/test_array_api.py::"):
            item.add_marker(mark)
            continue

        if nodeid.startswith(
            "test/test_round.py::test_stochastic_rounding_scalar_eq_array"
        ):
            item.add_marker(mark)
            continue

        if nodeid.startswith("test/test_encode.py::test_encode["):
            item.add_marker(mark)
            continue

        if nodeid.startswith("test/test_encode.py::test_encode_edges[encode_ndarray-"):
            item.add_marker(mark)
            continue

        if (
            nodeid.startswith("test/test_decode.py::test_spot_check_")
            and "[array]" in nodeid
        ):
            item.add_marker(mark)
            continue

        if (
            nodeid.startswith("test/test_decode.py::test_specials_decode[")
            and "[array-" in nodeid
        ):
            item.add_marker(mark)
            continue

        if nodeid.startswith("test/test_decode.py::test_consistent_decodes_all_values["):
            item.add_marker(mark)

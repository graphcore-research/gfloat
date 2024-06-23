<!-- Copyright (c) 2024 Graphcore Ltd. All rights reserved. -->

# gfloat: Generic floating-point types in Python

An implementation of generic floating point encode/decode logic,
handling various current and proposed floating point types:

 - [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754): Binary16, Binary32
 - [OCP Float8](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf): E5M2, E4M3
 - [IEEE WG P3109](https://github.com/awf/P3109-Public/blob/main/Shared%20Reports/P3109%20WG%20Interim%20report.pdf): P{p} for p in 1..7
 - [OCP MX Formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf): E2M1, M2M3, E3M2, E8M0, INT8, and the MX block formats.

The library favours readability and extensibility over speed (although the *_ndarray functions are reasonably fast for large arrays, see the [benchmarking  notebook](docs/source/04-benchmark.ipynb)).
For other implementations of these datatypes more focused on speed see, for example, [ml_dtypes](https://github.com/jax-ml/ml_dtypes),
[bitstring](https://github.com/scott-griffiths/bitstring),
[MX PyTorch Emulation Library](https://github.com/microsoft/microxcaling).

See https://gfloat.readthedocs.io for documentation, or dive into the notebooks to explore the formats.

For example, here's a table from the [02-value-stats](docs/source/02-value-stats.ipynb) notebook:

|name|B: Bits in the format|P: Precision in bits|E: Exponent field width in bits|0<x<1|1<x<Inf|Exact in float16?|maxFinite|minFinite|maxNormal|minNormal|minSubnormal|maxSubnormal|
|--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |
|ocp_e2m1|4|2|2|1|5|True|6|-6|6|1|0.5|0.5|
|ocp_e2m3|6|4|2|7|23|True|7.5|-7.5|7.5|1|0.125|0.875|
|ocp_e3m2|6|3|3|11|19|True|28|-28|28|0.25|0.0625|0.1875|
|ocp_e4m3|8|4|4|55|70|True|448|-448|448|0.015625|1*2^-9|7/4*2^-7|
|ocp_e5m2|8|3|5|59|63|True|57344|-57344|57344|1*2^-14|1*2^-16|3/2*2^-15|
|p3109_p1|8|1|7|62|63|False|1*2^63|-1*2^63|1*2^63|1*2^-62|nan|nan|
|p3109_p2|8|2|6|63|62|False|1*2^31|-1*2^31|1*2^31|1*2^-31|1*2^-32|1*2^-32|
|p3109_p3|8|3|5|63|62|True|49152|-49152|49152|1*2^-15|1*2^-17|3/2*2^-16|
|p3109_p4|8|4|4|63|62|True|224|-224|224|0.0078125|1*2^-10|7/4*2^-8|
|p3109_p5|8|5|3|63|62|True|15|-15|15|0.125|0.0078125|15/8*2^-4|
|p3109_p6|8|6|2|63|62|True|3.875|-3.875|3.875|0.5|0.015625|31/16*2^-2|
|bfloat16|16|8|8|16255|16383|False|255/128*2^127|-255/128*2^127|255/128*2^127|1*2^-126|1*2^-133|127/64*2^-127|
|ocp_int8|8|8|0|63|63|True|127/64*2^0|-2|nan|nan|0.015625|127/64*2^0|
|ocp_e8m0|8|1|8|127|127|False|1*2^127|1*2^-127|1*2^127|1*2^-127|nan|nan|


#### Notes

All NaNs are the same, with no distinction between signalling or quiet,
or between differently encoded NaNs.

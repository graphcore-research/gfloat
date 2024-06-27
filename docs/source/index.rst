.. Copyright (c) 2024 Graphcore Ltd. All rights reserved.

.. note::

  Check the version number of this documentation against the `gfloat` version
  you are using.  "Latest" refers to the head on https://github.com/graphcore-research/gfloat,
  while pypi versions installed using `pip install` will have corresponding `vX.Y.Z` tags.

GFloat: Generic floating point formats in Python
================================================

GFloat is designed to allow experimentation with a variety of floating-point
formats in Python.  Formats are parameterized by the primary IEEE-754 parameters
of:

  * Width in bits (k)
  * Precision (p)
  * Maximum exponent (emax)

with additional fields defining the presence/encoding of:

  * Infinities
  * Not-a-number (NaN) values
  * Negative zero
  * Subnormal numbers
  * Signed/unsigned
  * Two's complement encoding (of the significand)

This allows an implementation of generic floating point encode/decode logic,
handling various current and proposed floating point types:

 - `IEEE 754 <https://en.wikipedia.org/wiki/IEEE_754>`_: Binary16, Binary32
 - `Brain floating point <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_: BFloat16
 - |ocp_link|: E5M2, E4M3
 - |p3109_link|: P{p} for p in 1..7
 - Types from the |ocp_mx_link| spec: E8M0, INT8, and FP4, FP6 types

As well as block formats from |ocp_mx_link|.

.. |ocp_mx_link| raw:: html

   <a href="https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf" target="_blank">
     OCP MX
   </a>

.. |ocp_link| raw:: html

   <a href="https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf" target="_blank">
     OCP Float8
   </a>

.. |p3109_link| raw:: html

    <a href="https://raw.githubusercontent.com/P3109/Public/main/Shared%20Reports/P3109%20WG%20Interim%20Report.pdf" target="_blank">
      IEEE P3109
    </a>

Supported rounding modes include:

  * Directed modes: Toward Zero, Toward Positive, Toward Negative
  * Round-to-nearest, with Ties to Even or Ties to Away
  * Stochastic rounding, with specified numbers of random bits


Example
-------
This table (from example notebook :doc:`value-stats <02-value-stats>`) shows how
gfloat has been used to tabulate properties of various floating point formats.

 - name: Format
 - B: Bits in the format
 - P: Precision in bits
 - E: Exponent field width in bits
 - smallest: Smallest positive value
 - smallest_normal: Smallest positive normal value, n/a if no finite values are normal
 - max: Largest finite value
 - num_nans: Number of NaN values
 - num_infs: Number of infinities (2 or 0)

========  ===  ===  ===  ===========  =================  ============  ===========  ======
name        B    P    E  smallest     smallest_normal    max           num_nans       infs
========  ===  ===  ===  ===========  =================  ============  ===========  ======
ocp_e2m1    4    2    2  0.5          1                  6             0                 0
ocp_e2m3    6    4    2  0.125        1                  7.5           0                 0
ocp_e3m2    6    3    3  0.0625       0.25               28            0                 0
ocp_e4m3    8    4    4  ≈0.0019531   0.015625           448           2                 0
ocp_e5m2    8    3    5  ≈1.5259e-05  ≈6.1035e-05        57344         6                 2
p3109_p1    8    1    7  ≈2.1684e-19  ≈2.1684e-19        ≈9.2234e+18   1                 2
p3109_p2    8    2    6  ≈2.3283e-10  ≈4.6566e-10        ≈2.1475e+09   1                 2
p3109_p3    8    3    5  ≈7.6294e-06  ≈3.0518e-05        49152         1                 2
p3109_p4    8    4    4  ≈0.00097656  0.0078125          224           1                 2
p3109_p5    8    5    3  0.0078125    0.125              15            1                 2
p3109_p6    8    6    2  0.015625     0.5                3.875         1                 2
binary16   16   11    5  ≈5.9605e-08  ≈6.1035e-05        65504         2046              2
bfloat16   16    8    8  ≈9.1835e-41  ≈1.1755e-38        ≈3.3895e+38   254               2
binary32   32   24    8  ≈1.4013e-45  ≈1.1755e-38        ≈3.4028e+38   ≈1.6777e+07       2
binary64   64   53   11  4.9407e-324  ≈2.2251e-308       ≈1.7977e+308  ≈9.0072e+15       2
ocp_e8m0    8    1    8  ≈5.8775e-39  ≈5.8775e-39        ≈1.7014e+38   1                 0
ocp_int8    8    8    0  0.015625     n/a                ≈  1.9844     0                 0
========  ===  ===  ===  ===========  =================  ============  ===========  ======

In the above table, values which are not exact are indicated with the "≈" symbol.
And here's the same table, but with values which don't render exactly as short floats
printed as rationals times powers of 2:

========  ===  ===  ===  ===========  =================  ========================================  ======================================  ======
name        B    P    E  smallest     smallest_normal    max                                       num_nans                                  infs
========  ===  ===  ===  ===========  =================  ========================================  ======================================  ======
ocp_e2m1    4    2    2  0.5          1                  6                                         0                                            0
ocp_e2m3    6    4    2  0.125        1                  7.5                                       0                                            0
ocp_e3m2    6    3    3  0.0625       0.25               28                                        0                                            0
ocp_e4m3    8    4    4  2^-9         0.015625           448                                       2                                            0
ocp_e5m2    8    3    5  2^-16        2^-14              57344                                     6                                            2
p3109_p1    8    1    7  2^-62        2^-62              2^63                                      1                                            2
p3109_p2    8    2    6  2^-32        2^-31              2^31                                      1                                            2
p3109_p3    8    3    5  2^-17        2^-15              49152                                     1                                            2
p3109_p4    8    4    4  2^-10        0.0078125          224                                       1                                            2
p3109_p5    8    5    3  0.0078125    0.125              15                                        1                                            2
p3109_p6    8    6    2  0.015625     0.5                3.875                                     1                                            2
binary16   16   11    5  2^-24        2^-14              65504                                     2046                                         2
bfloat16   16    8    8  2^-133       2^-126             255/128*2^127                             254                                          2
binary32   32   24    8  2^-149       2^-126             16777215/8388608*2^127                    8388607/4194304*2^23                         2
binary64   64   53   11  4.9407e-324  2^-1022            9007199254740991/9007199254740992*2^1024  4503599627370495/4503599627370496*2^53       2
ocp_e8m0    8    1    8  2^-127       2^-127             2^127                                     1                                            0
ocp_int8    8    8    0  0.015625     n/a                127/64*2^0                                0                                            0
========  ===  ===  ===  ===========  =================  ========================================  ======================================  ======


See Also
--------

GFloat, being a pure Python library, favours readability and extensibility over speed
(although the `*_ndarray` functions are reasonably fast for large arrays).
For fast implementations of these datatypes see, for example,
`ml_dtypes <https://github.com/jax-ml/ml_dtypes>`_,
`bitstring <https://github.com/scott-griffiths/bitstring>`_,
`MX PyTorch Emulation Library <https://github.com/microsoft/microxcaling>`_,
`APyTypes <https://apytypes.github.io/apytypes>`_.

To get started with the library, we recommend perusing the notebooks,
otherwise you may wish to jump straight into the API.

Contents
========

.. toctree::
   :hidden:

   self

.. toctree::

   notebooks
   api
   formats


Index and Search
================

* :ref:`genindex`
* :ref:`search`

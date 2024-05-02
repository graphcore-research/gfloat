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

with additional fields defining the encoding of infinities, Not-a-number (NaN) values,
and negative zero, among others (see :class:`gfloat.FormatInfo`.)

This allows an implementation of generic floating point encode/decode logic,
handling various current and proposed floating point types:

 - `IEEE 754 <https://en.wikipedia.org/wiki/IEEE_754>`_: Binary16, Binary32
 - `OCP Float8 <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf>`_: E5M2, E4M3, and MX formats
 - `IEEE WG P3109 <https://github.com/awf/P3109-Public/blob/main/Shared%20Reports/P3109%20WG%20Interim%20report.pdf>`_: P{p} for p in 1..7

The library favours readability and extensibility over speed - for fast
implementations of these datatypes see, for example,
`ml_dtypes <https://github.com/jax-ml/ml_dtypes>`_,
`bitstring <https://github.com/scott-griffiths/bitstring>`_,
`MX PyTorch Emulation Library <https://github.com/microsoft/microxcaling>`_.

To get started with the library, we recommend perusing the notebooks,
otherwise you may wish to jump straight into the API.

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

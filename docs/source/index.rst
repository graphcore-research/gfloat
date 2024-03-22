
GFloat: Generic floating point formats in Python
================================================


.. toctree::
   :maxdepth: 2
   :caption: Contents:

GFloat is designed to allow experimentation with a variety of floating-point
formats in Python.  Formats are parameterized by the primary IEEE-754 parameters
of:

  * Width in bits (k)
  * Precision (p)
  * Maximum exponent (emax)

with additional fields defining the encoding of infinities, Not-a-number (NaN) values, 
and negative zero.

This allows an implementation of generic floating point encode/decode logic,
handling various current and proposed floating point types:

 - `IEEE 754 <https://en.wikipedia.org/wiki/IEEE_754>`_: Binary16, Binary32
 - `OCP Float8 <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf>`_: E5M2, E4M3
 - `IEEE WG P3109 <https://github.com/awf/P3109-Public/blob/main/Shared%20Reports/P3109%20WG%20Interim%20report.pdf>`_: P{p} for p in 1..7


API
===

.. module:: gfloat

.. autofunction:: decode_float
.. autofunction:: round_float
.. autoclass:: FormatInfo()
   :members:
.. autoclass:: FloatClass()
   :members:
.. autoclass:: RoundMode()
   :members:
.. autoclass:: FloatValue()
   :members:

Defined Formats
===============

.. module:: gfloat.formats

.. autodata:: format_info_binary32
.. autodata:: format_info_binary16
.. autodata:: format_info_bfloat16
.. autodata:: format_info_ocp_e5m2
.. autodata:: format_info_ocp_e4m3
.. autofunction:: format_info_p3109

Index and Search
================

* :ref:`genindex`
* :ref:`search`

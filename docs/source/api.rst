.. Copyright (c) 2024 Graphcore Ltd. All rights reserved.

API
===

.. module:: gfloat

Scalar Functions
----------------

.. autofunction:: decode_float
.. autofunction:: round_float
.. autofunction:: encode_float

Block format functions
----------------------

.. autofunction:: decode_block
.. autofunction:: encode_block
.. autofunction:: quantize_block

.. autofunction:: compute_scale_amax


Classes
-------

.. autoclass:: FormatInfo()
   :members:
.. autoclass:: FloatClass()
   :members:
.. autoclass:: RoundMode()
   :members:
.. autoclass:: FloatValue()
   :members:
.. autoclass:: BlockFormatInfo()
   :members:

Pretty printers
---------------

.. autofunction:: float_pow2str
.. autofunction:: float_tilde_unless_roundtrip_str

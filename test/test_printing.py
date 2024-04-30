from gfloat import float_pow2str


def test_pow2str():
    assert float_pow2str(127) == "127/64*2^6"
    assert float_pow2str(1.0625 * 2.0**-12) == "17/16*2^-12"
    assert float_pow2str(3.0 * 2.0**-12) == "3/2*2^-11"
    assert float_pow2str(3.0 / 16 * 2.0**-8) == "3/2*2^-11"
    assert float_pow2str(3.0 / 16 * 2.0**-8, min_exponent=-8) == "3/16*2^-8"

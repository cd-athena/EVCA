"""Hasler & Suesstrunk M^(3) colorfulness, computed natively on YUV chroma planes."""
import numpy as np
import pytest

from libs.colorfulness import calculate_hasler_suesstrunk_colorfulness_yuv as colorfulness

CH, CW = 1080 // 2, 1920 // 2


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        colorfulness(np.zeros((CH, CW), np.uint8), np.zeros((CH, CW + 2), np.uint8))


@pytest.mark.parametrize('bit_depth', [8, 10])
def test_grayscale_is_zero(bit_depth):
    """U = V = the neutral point carries no color at any bit depth."""
    neutral = 1 << (bit_depth - 1)
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    plane = np.full((CH, CW), neutral, dtype=dtype)
    assert colorfulness(plane, plane, bit_depth=bit_depth) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize('bit_depth', [8, 10])
def test_solid_color_isolates_the_mean_component(bit_depth):
    """A saturated solid color has zero variance, so only the 0.3 * mu term survives."""
    maxval = (1 << bit_depth) - 1
    offset = maxval - (1 << (bit_depth - 1))
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    plane = np.full((CH, CW), maxval, dtype=dtype)
    expected = 0.3 * np.sqrt(2 * offset ** 2) * 2.05
    assert colorfulness(plane, plane, bit_depth=bit_depth) == pytest.approx(expected, abs=1e-4)


def test_checkerboard_isolates_the_variance_component():
    """A 50/50 split of 0 and 255 has sigma 127.5 per channel and a mean of 127.5,
    i.e. 0.5 off neutral -- so the variance term dominates by a factor of ~850."""
    plane = np.zeros((CH, CW), np.uint8)
    plane[:, :CW // 2] = 255
    sigma, mu = np.sqrt(2 * 127.5 ** 2), np.sqrt(2 * 0.5 ** 2)
    expected = (sigma + 0.3 * mu) * 2.05
    assert colorfulness(plane, plane, bit_depth=8) == pytest.approx(expected, abs=1e-4)

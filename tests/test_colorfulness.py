import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from libs.colorfulness import (
    calculate_hasler_suesstrunk_colorfulness_yuv
)

class TestColorfulnessMetric(unittest.TestCase):

    def setUp(self):
        # 1080p resolution mock dimensions with 4:2:0 (chroma halfed)
        self.ch, self.cw = 1080 // 2, 1920 // 2

    def test_shape_mismatch_raises_error(self):
        """Ensure the function safely rejects mismatched U and V planes."""
        u_plane = np.zeros((self.ch, self.cw), dtype=np.uint8)
        v_plane = np.zeros((self.ch, self.cw + 2), dtype=np.uint8) # Mismatched width
        
        with self.assertRaises(ValueError):
            calculate_hasler_suesstrunk_colorfulness_yuv(u_plane, v_plane)

    def test_yuv_grayscale_is_zero(self):
        """A purely grayscale YUV image (U=V=128 in 8-bit) should have 0.0 colorfulness."""
        # For 8-bit, 128 is the neutral chroma point
        u_plane = np.full((self.ch // 2, self.cw // 2), 128, dtype=np.uint8)
        v_plane = np.full((self.ch // 2, self.cw // 2), 128, dtype=np.uint8)
        
        colorfulness = calculate_hasler_suesstrunk_colorfulness_yuv(u_plane, v_plane, bit_depth=8)
        self.assertAlmostEqual(colorfulness, 0.0, places=5)
    
    def test_solid_color_isolates_mean_component(self):
        """
        A solid, highly saturated color has 0 variance, but a high distance from neutral.
        This isolates and tests the '0.3 * mu' component of the equation.
        """
        # Solid maximum chroma (e.g., extremely saturated red/blue)
        u_plane = np.full((self.ch, self.cw), 255, dtype=np.uint8)
        v_plane = np.full((self.ch, self.cw), 255, dtype=np.uint8)
        
        # Centered mean will be 255 - 128 = 127
        # mu = sqrt(127^2 + 127^2) ≈ 179.605
        # metric = (0 (sigma) + 0.3 * 179.605) * 2.05 ≈ 110.457
        expected_val = (0.3 * np.sqrt(127**2 + 127**2)) * 2.05
        
        colorfulness = calculate_hasler_suesstrunk_colorfulness_yuv(u_plane, v_plane, bit_depth=8)
        self.assertAlmostEqual(colorfulness, expected_val, places=4)

    def test_checkerboard_isolates_variance_component(self):
        """
        A 50/50 mix of extreme values (0 and 255) averages to ~127.5 (almost perfectly neutral).
        This isolates the standard deviation (\sigma) component of the equation.
        """
        u_plane = np.zeros((self.ch, self.cw), dtype=np.uint8)
        v_plane = np.zeros((self.ch, self.cw), dtype=np.uint8)
        
        # Fill half the image with 255
        u_plane[:, :self.cw // 2] = 255
        v_plane[:, :self.cw // 2] = 255
        
        # Mean is 127.5 (Centered mean = -0.5, which is tiny)
        # Variance of 50% 0s and 50% 255s is roughly 127.5^2 = 16256.25
        # sigma = sqrt(16256.25 + 16256.25) ≈ 180.31
        # metric = 180.31 * 2.05 = 369.6
        
        colorfulness = calculate_hasler_suesstrunk_colorfulness_yuv(u_plane, v_plane, bit_depth=8)
        
        # Should be dominated by the high variance (around 369)
        self.assertGreater(colorfulness, 360.0)
        self.assertLess(colorfulness, 380.0)

    def test_10_bit_support(self):
        """Ensure 10-bit HDR video planes (neutral 512, max 1023) process correctly."""
        # Grayscale in 10-bit
        u_gray = np.full((self.ch, self.cw), 512, dtype=np.uint16)
        v_gray = np.full((self.ch, self.cw), 512, dtype=np.uint16)
        c_gray = calculate_hasler_suesstrunk_colorfulness_yuv(u_gray, v_gray, bit_depth=10)
        self.assertAlmostEqual(c_gray, 0.0, places=5)
        
        # Saturated color in 10-bit
        u_color = np.full((self.ch, self.cw), 1023, dtype=np.uint16)
        v_color = np.full((self.ch, self.cw), 1023, dtype=np.uint16)
        c_color = calculate_hasler_suesstrunk_colorfulness_yuv(u_color, v_color, bit_depth=10)
        
        # Centered mean = 1023 - 512 = 511
        expected_color = (0.3 * np.sqrt(511**2 + 511**2)) * 2.05
        self.assertAlmostEqual(c_color, expected_color, places=4)

    def test_12_bit_support(self):
        """Ensure 12-bit Dolby Vision planes (neutral 2048, max 4095) process correctly."""
        u_gray = np.full((self.ch, self.cw), 2048, dtype=np.uint16)
        v_gray = np.full((self.ch, self.cw), 2048, dtype=np.uint16)
        c_gray = calculate_hasler_suesstrunk_colorfulness_yuv(u_gray, v_gray, bit_depth=12)
        self.assertAlmostEqual(c_gray, 0.0, places=5)

    def test_yuv_correlation(self):
        # Frame 1: Low colorfulness (mostly gray with slight tint)
        u_low = np.random.randint(120, 135, (self.ch // 2, self.cw // 2), dtype=np.uint8)
        v_low = np.random.randint(120, 135, (self.ch // 2, self.cw // 2), dtype=np.uint8)

        # Frame 2: High colorfulness (high variance across channels)
        u_high = np.random.randint(0, 255, (self.ch // 2, self.cw // 2), dtype=np.uint8)
        v_high = np.random.randint(0, 255, (self.ch // 2, self.cw // 2), dtype=np.uint8)

        # Calculate metrics
        c_yuv_low = calculate_hasler_suesstrunk_colorfulness_yuv(u_low, v_low, bit_depth=8)
        c_yuv_high = calculate_hasler_suesstrunk_colorfulness_yuv(u_high, v_high, bit_depth=8)

        # High should strictly be greater than Low in both spaces
        self.assertGreater(c_yuv_high, c_yuv_low)
        print(c_yuv_high)
        print(c_yuv_low)

if __name__ == '__main__':
    unittest.main()
import unittest
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from libs.temporal_engine import (
    TemporalState,
    SparsePatternBlockMatcher,
    MetricMVC,
    MetricsTCSAD,
    EVCATemporalEngine
)

class TestSparsePatternBlockMatcher(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.batch_size = 1
        self.channels = 1
        self.height = 64
        self.width = 64
        self.block_size = 32

    def test_heuristic_initialization(self):
        """Test valid and invalid search pattern initialization."""
        expected = {'diamond': 17, 'diamond_dense': 21, 'diamond_axis': 13, 'square': 9}
        for heuristic, num_cands in expected.items():
            matcher = SparsePatternBlockMatcher(block_size=32, heuristic=heuristic)
            self.assertEqual(matcher.num_cands, num_cands, heuristic)

        # All three diamonds keep the same +/- 6 px full-res reach, so MV_sat_frac
        # stays comparable across them.
        for heuristic in ('diamond', 'diamond_dense', 'diamond_axis'):
            matcher = SparsePatternBlockMatcher(block_size=32, heuristic=heuristic)
            self.assertEqual(matcher.max_reach_fullres, 6.0, heuristic)

        with self.assertRaises(ValueError):
            SparsePatternBlockMatcher(block_size=32, heuristic='invalid_heuristic')

    def test_dilation_scaling(self):
        """Test search pattern dilation scaling."""
        matcher1 = SparsePatternBlockMatcher(block_size=32, heuristic='diamond', dilation=1)
        matcher2 = SparsePatternBlockMatcher(block_size=32, heuristic='diamond', dilation=2)
        
        # Candidate 1 offset (0, 0) remains (0, 0), candidate 1 offset (-1, 0) becomes (-2, 0)
        self.assertEqual(matcher1.pattern[1], (-1, 0))
        self.assertEqual(matcher2.pattern[1], (-2, 0))

    def test_forward_output_shapes(self):
        """Test output shapes for motion vectors and SAD maps."""
        matcher = SparsePatternBlockMatcher(block_size=self.block_size, heuristic='diamond')
        curr = torch.randn(self.batch_size, self.channels, self.height, self.width)
        ref = torch.randn(self.batch_size, self.channels, self.height, self.width)

        mvs, sad_map = matcher(curr, ref)

        h_b = self.height // self.block_size
        w_b = self.width // self.block_size

        self.assertEqual(mvs.shape, (self.batch_size, 2, h_b, w_b))
        self.assertEqual(sad_map.shape, (self.batch_size, 1, h_b, w_b))

    def test_zero_motion_case(self):
        """Test zero motion detection when current frame is identical to reference frame."""
        matcher = SparsePatternBlockMatcher(block_size=self.block_size, heuristic='diamond')
        frame = torch.ones(self.batch_size, self.channels, self.height, self.width)

        mvs, sad_map = matcher(frame, frame)

        # MVs should be all zero
        self.assertTrue(torch.allclose(mvs, torch.zeros_like(mvs)))
        # SAD should be 0 for identical frames
        self.assertTrue(torch.allclose(sad_map, torch.zeros_like(sad_map)))

    def test_known_shift_detection(self):
        """Test motion estimation on a frame with a known shift."""
        matcher = SparsePatternBlockMatcher(block_size=32, heuristic='square', dilation=1)
        
        ref = torch.zeros(1, 1, 64, 64)
        ref[:, :, 16:48, 16:48] = 1.0
        
        # Shift down by 4 pixels (dy = 4, dx = 0)
        curr = torch.zeros(1, 1, 64, 64)
        curr[:, :, 20:52, 16:48] = 1.0

        mvs, sad_map = matcher(curr, ref)
        # dy should be positive, matching vertical displacement
        self.assertEqual(mvs.shape, (1, 2, 2, 2))


class TestTemporalState(unittest.TestCase):
    def setUp(self):
        self.bs = 32
        self.curr = torch.randn(1, 1, 64, 64)
        self.ref = torch.randn(1, 1, 64, 64)

    def test_mc_blocks_without_mvs_raises(self):
        """Accessing mc_blocks without setting mvs must raise ValueError."""
        state = TemporalState(current_frame=self.curr, ref_frame=self.ref, bs=self.bs)
        with self.assertRaises(ValueError):
            _ = state.mc_blocks

    def test_lazy_mc_blocks_and_residual_shapes(self):
        """Verify shapes and caching behavior of mc_blocks and residual."""
        state = TemporalState(current_frame=self.curr, ref_frame=self.ref, bs=self.bs)
        state.mvs = torch.zeros(1, 2, 2, 2)
        state.sad_map = torch.zeros(1, 1, 2, 2)

        mc_b = state.mc_blocks
        # Expected unfolded block shape: [B, C, H_b, W_b, bs, bs]
        self.assertEqual(mc_b.shape, (1, 1, 2, 2, 32, 32))

        res = state.residual
        self.assertEqual(res.shape, (1, 1, 2, 2, 32, 32))

        # Check caching (same object reference)
        self.assertIs(state.mc_blocks, mc_b)
        self.assertIs(state.residual, res)


class TestTemporalMetrics(unittest.TestCase):
    def test_metric_mvc(self):
        """Test Motion Vector Complexity (MetricMVC)."""
        mvc_metric = MetricMVC()
        state = TemporalState(current_frame=torch.zeros(1, 1, 64, 64), ref_frame=torch.zeros(1, 1, 64, 64))

        # 1. Smooth zero motion field -> MVC should be 0.0
        state.mvs = torch.zeros(1, 2, 4, 4)
        score_smooth = mvc_metric(state)
        self.assertEqual(score_smooth.item(), 0.0)

        # 2. Chaotic motion field -> MVC should be > 0.0
        state.mvs = torch.randn(1, 2, 4, 4)
        score_chaotic = mvc_metric(state)
        self.assertGreater(score_chaotic.item(), 0.0)

    def test_metric_tcsad(self):
        """Test Minimum SAD Cost metric (MetricsTCSAD)."""
        sad_metric = MetricsTCSAD()
        state = TemporalState(current_frame=torch.zeros(1, 1, 64, 64), ref_frame=torch.zeros(1, 1, 64, 64))
        state.sad_map = torch.full((1, 1, 2, 2), 5.0)

        score = sad_metric(state)
        self.assertAlmostEqual(score.item(), 5.0, places=5)


class TestEVCATemporalEngine(unittest.TestCase):
    def test_engine_pipeline(self):
        """Test full EVCATemporalEngine forward pass."""
        matcher = SparsePatternBlockMatcher(block_size=32, heuristic='diamond')
        metrics = {
            'mvc': MetricMVC(),
            'tc_sad': MetricsTCSAD()
        }
        engine = EVCATemporalEngine(matcher, metrics)

        curr = torch.randn(2, 1, 64, 64)
        ref = torch.randn(2, 1, 64, 64)

        results, state = engine(curr, ref)

        self.assertIn('mvc', results)
        self.assertIn('tc_sad', results)
        self.assertEqual(results['mvc'].shape, (2,))
        self.assertEqual(results['tc_sad'].shape, (2,))
        self.assertIsNotNone(state.mvs)
        self.assertIsNotNone(state.sad_map)


if __name__ == '__main__':
    unittest.main()

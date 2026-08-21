"""ME endpoint error and MC residual checks on synthetic sequences (Phases 0.1, 1.4)."""
import numpy as np
import pytest
import torch

from libs.temporal_engine import SparsePatternBlockMatcher, TemporalState
from validation.synthetic import (gen_translation, gen_static_noise, gen_cut,
                                  make_base_texture)


def _to_batch(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(frame)).unsqueeze(0).unsqueeze(0).float()


def _epe(mvs: torch.Tensor, gt_dy: float, gt_dx: float) -> float:
    """Mean endpoint error over interior blocks (borders see padding artifacts)."""
    dy = mvs[0, 0, 1:-1, 1:-1]
    dx = mvs[0, 1, 1:-1, 1:-1]
    return torch.sqrt((dy - gt_dy) ** 2 + (dx - gt_dx) ** 2).mean().item()


@pytest.mark.parametrize('vy,vx', [(0, 2), (0, 4), (0, 6), (-4, 0), (2, 0)])
def test_diamond_epe_in_range(vy, vx):
    """Even axis-aligned shifts within +/-6 px must be matched exactly."""
    frames, _ = gen_translation(160, 224, 3, vy=vy, vx=vx, seed=5)
    matcher = SparsePatternBlockMatcher(block_size=32, heuristic='diamond')
    mvs, _ = matcher(_to_batch(frames[2]), _to_batch(frames[1]))
    assert _epe(mvs, vy, vx) < 1e-6


def test_diamond_epe_odd_shift():
    """Odd shifts are off-grid for the half-res search; best reachable EPE is 1 px."""
    frames, _ = gen_translation(160, 224, 3, vy=0, vx=5, seed=6)
    matcher = SparsePatternBlockMatcher(block_size=32, heuristic='diamond')
    mvs, _ = matcher(_to_batch(frames[2]), _to_batch(frames[1]))
    assert _epe(mvs, 0, 5) <= 1.0 + 1e-6


def test_square_epe_diagonal():
    """The square pattern must match its diagonal candidates exactly."""
    frames, _ = gen_translation(160, 224, 3, vy=4, vx=-4, seed=7)
    matcher = SparsePatternBlockMatcher(block_size=32, heuristic='square')
    mvs, _ = matcher(_to_batch(frames[2]), _to_batch(frames[1]))
    assert _epe(mvs, 4, -4) < 1e-6


def test_static_noise_zero_motion():
    frames, _ = gen_static_noise(128, 160, 3, sigma=3.0, seed=8)
    matcher = SparsePatternBlockMatcher(block_size=32, heuristic='diamond')
    mvs, _ = matcher(_to_batch(frames[2]), _to_batch(frames[1]))
    assert _epe(mvs, 0, 0) < 1e-6


def test_mc_residual_near_zero_for_integer_translation():
    """Phase 0.1 regression: with align_corners=False the MV field is registered to
    block centers, so an in-range global integer translation must produce a
    near-zero motion-compensated residual on interior blocks."""
    frames, _ = gen_translation(160, 224, 3, vy=0, vx=4, seed=9)
    curr, ref = _to_batch(frames[2]), _to_batch(frames[1])
    matcher = SparsePatternBlockMatcher(block_size=32, heuristic='diamond')
    state = TemporalState(current_frame=curr, ref_frame=ref, bs=32)
    state.mvs, state.sad_map = matcher(curr, ref)

    residual = state.residual  # [B, C, H_b, W_b, bs, bs]
    interior = residual[:, :, 1:-1, 1:-1]
    uncompensated = (curr - ref).abs().mean().item()
    mc_err = interior.abs().mean().item()
    assert mc_err < 1e-3, f'interior MC residual {mc_err} not near zero'
    assert mc_err < uncompensated * 1e-3


def test_cut_produces_high_sad():
    """Across a hard cut the minimum SAD must stay comparable to frame difference."""
    frames, gt = gen_cut(128, 160, 4, seed=10)
    c = gt['cut_frame']
    matcher = SparsePatternBlockMatcher(block_size=32, heuristic='diamond')
    _, sad_within = matcher(_to_batch(frames[c + 1]), _to_batch(frames[c]))
    _, sad_across = matcher(_to_batch(frames[c]), _to_batch(frames[c - 1]))
    assert sad_across.mean().item() > 10 * max(sad_within.mean().item(), 1e-6)


def test_texture_has_unique_sad_minimum():
    """Sanity: the band-pass texture must give SAD a unique minimum at zero shift."""
    tex = make_base_texture(96, 128, seed=11)
    t = _to_batch(tex)
    matcher = SparsePatternBlockMatcher(block_size=32, heuristic='diamond')
    mvs, _ = matcher(t, t)
    assert mvs.abs().max().item() == 0.0

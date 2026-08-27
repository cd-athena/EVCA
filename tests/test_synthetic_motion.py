"""ME endpoint error and MC residual checks on synthetic sequences."""
import numpy as np
import pytest
import torch

from libs.temporal_engine import PatternBlockMatcher, TemporalState
from validation.synthetic import (gen_translation, gen_static_noise, gen_cut,
                                  make_base_texture)

OFFSET = 2      # CLI default for --me-offset


def _to_batch(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(frame)).unsqueeze(0).unsqueeze(0).float()


def _epe(mvs: torch.Tensor, gt_dy: float, gt_dx: float) -> float:
    """Mean endpoint error over interior blocks (borders see padding artifacts)."""
    dy = mvs[0, 0, 1:-1, 1:-1]
    dx = mvs[0, 1, 1:-1, 1:-1]
    return torch.sqrt((dy - gt_dy) ** 2 + (dx - gt_dx) ** 2).mean().item()


@pytest.mark.parametrize('vy,vx', [(0, OFFSET), (0, -OFFSET), (OFFSET, 0), (-OFFSET, 0)])
def test_diamond_epe_on_axis_shifts(vy, vx):
    """The diamond's four candidates must match their own shifts exactly."""
    frames, _ = gen_translation(160, 224, 3, vy=vy, vx=vx, seed=5)
    matcher = PatternBlockMatcher(32, 'diamond', offset=OFFSET)
    mvs, _ = matcher(_to_batch(frames[2]), _to_batch(frames[1]))
    assert _epe(mvs, vy, vx) < 1e-6


@pytest.mark.parametrize('offset', [1, 3, 5])
def test_odd_shifts_are_exactly_representable_at_full_resolution(offset):
    """At pool 1 every integer motion vector is exact, odd ones included."""
    frames, _ = gen_translation(160, 224, 3, vy=0, vx=offset, seed=6)
    matcher = PatternBlockMatcher(32, 'diamond', offset=offset)
    mvs, _ = matcher(_to_batch(frames[2]), _to_batch(frames[1]))
    assert _epe(mvs, 0, offset) < 1e-6


@pytest.mark.parametrize('pool,shift,exact', [
    (2, 4, True),    # a multiple of the pooling factor
    (2, 2, True),
    (2, 3, False),   # falls between grid points
    (4, 8, True),
    (4, 6, False),
])
def test_pooling_quantises_vectors_to_a_pool_sized_grid(pool, shift, exact):
    """The precision the pooled search trades away, stated as a bound.

    Searching on a 1/P image can only emit vectors that are multiples of P, so a shift
    that is a multiple of P stays exact and any other carries at most P/2 px of
    endpoint error. This is the whole cost of pooling the search, and it is why
    --me-offset stays denominated in full-resolution pixels.
    """
    frames, _ = gen_translation(160, 224, 3, vy=0, vx=shift, seed=6)
    matcher = PatternBlockMatcher(32, 'diamond', offset=shift, pool=pool)
    mvs, _ = matcher(_to_batch(frames[2]), _to_batch(frames[1]))
    err = _epe(mvs, 0, shift)
    assert (mvs % pool == 0).all(), 'pooled search emitted an off-grid vector'
    if exact:
        assert err < 1e-6
    else:
        assert 0 < err <= pool / 2 + 1e-6


def test_pooled_search_tracks_motion_the_full_res_pattern_cannot_reach():
    """The point of pooling: the same five candidates cover four times the ground.

    A 12 px pan is far outside a +/-3 px full-resolution pattern, but a pool-4 search
    with the same candidate count reaches it exactly.
    """
    frames, _ = gen_translation(160, 224, 3, vy=0, vx=12, seed=12)
    curr, ref = _to_batch(frames[2]), _to_batch(frames[1])
    near = PatternBlockMatcher(32, 'diamond', offset=3, pool=1)(curr, ref)[0]
    far = PatternBlockMatcher(32, 'diamond', offset=12, pool=4)(curr, ref)[0]
    assert _epe(near, 0, 12) > 8.0
    assert _epe(far, 0, 12) < 1e-6


def test_square_epe_diagonal():
    """The square pattern must match its diagonal candidates exactly."""
    frames, _ = gen_translation(160, 224, 3, vy=3, vx=-3, seed=7)
    matcher = PatternBlockMatcher(32, 'square', offset=3)
    mvs, _ = matcher(_to_batch(frames[2]), _to_batch(frames[1]))
    assert _epe(mvs, 3, -3) < 1e-6


def test_diamond_cannot_reach_diagonal_motion():
    """The diamond is axis-only by construction; the square is the diagonal option.

    Recorded so the trade-off between the two shapes stays visible: a 45-degree pan is
    exactly the case where the choice of pattern matters.
    """
    frames, _ = gen_translation(160, 224, 3, vy=2, vx=2, seed=7)
    curr, ref = _to_batch(frames[2]), _to_batch(frames[1])
    diamond = PatternBlockMatcher(32, 'diamond', offset=2)(curr, ref)[0]
    square = PatternBlockMatcher(32, 'square', offset=2)(curr, ref)[0]
    assert _epe(square, 2, 2) < 1e-6
    assert _epe(diamond, 2, 2) > 1.0


def test_static_noise_zero_motion():
    frames, _ = gen_static_noise(128, 160, 3, sigma=3.0, seed=8)
    matcher = PatternBlockMatcher(32, 'diamond', offset=OFFSET)
    mvs, _ = matcher(_to_batch(frames[2]), _to_batch(frames[1]))
    assert _epe(mvs, 0, 0) < 1e-6


def test_mc_residual_near_zero_for_integer_translation():
    """With align_corners=False the MV field is registered to block centers, so an
    in-range global integer translation must produce a near-zero motion-compensated
    residual on interior blocks."""
    frames, _ = gen_translation(160, 224, 3, vy=0, vx=OFFSET, seed=9)
    curr, ref = _to_batch(frames[2]), _to_batch(frames[1])
    matcher = PatternBlockMatcher(32, 'diamond', offset=OFFSET)
    state = TemporalState(current_frame=curr, ref_frame=ref, bs=32)
    state.mvs, state.sad_map = matcher(curr, ref)

    interior = state.residual[:, :, 1:-1, 1:-1]     # [B, C, H_b, W_b, bs, bs]
    uncompensated = (curr - ref).abs().mean().item()
    mc_err = interior.abs().mean().item()
    assert mc_err < 1e-3, f'interior MC residual {mc_err} not near zero'
    assert mc_err < uncompensated * 1e-3


def test_cut_produces_high_sad():
    """Across a hard cut the minimum SAD must stay comparable to frame difference."""
    frames, gt = gen_cut(128, 160, 4, seed=10)
    c = gt['cut_frame']
    matcher = PatternBlockMatcher(32, 'diamond', offset=OFFSET)
    _, sad_within = matcher(_to_batch(frames[c + 1]), _to_batch(frames[c]))
    _, sad_across = matcher(_to_batch(frames[c]), _to_batch(frames[c - 1]))
    assert sad_across.mean().item() > 10 * max(sad_within.mean().item(), 1e-6)


def test_texture_has_unique_sad_minimum():
    """Sanity: the band-pass texture must give SAD a unique minimum at zero shift."""
    t = _to_batch(make_base_texture(96, 128, seed=11))
    matcher = PatternBlockMatcher(32, 'diamond', offset=OFFSET)
    mvs, _ = matcher(t, t)
    assert mvs.abs().max().item() == 0.0

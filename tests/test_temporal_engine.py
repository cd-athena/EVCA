"""Motion estimator, TemporalState laziness, and the metric plugins."""
import pytest
import torch

from libs.temporal_engine import (EVCATemporalEngine, MetricMVC, MetricsTCSAD,
                                  PATTERN_SHAPES, SPARSE_PATTERN_SHAPES,
                                  PatternBlockMatcher, TemporalState, pooled_pair,
                                  quantise_offset, search_pattern)

BS, H, W = 32, 64, 64


# ----------------------------------------------------------------- search pattern

@pytest.mark.parametrize('shape', SPARSE_PATTERN_SHAPES)
@pytest.mark.parametrize('offset', [1, 2, 3, 7])
def test_pattern_is_five_unique_centred_points(shape, offset):
    pattern = search_pattern(shape, offset)
    assert len(pattern) == 5
    assert len(set(pattern)) == 5, 'duplicate candidate'
    assert (0, 0) in pattern, 'pattern must be able to report zero motion'
    # Every neighbour sits exactly `offset` from the centre along each axis it uses.
    neighbours = [c for c in pattern if c != (0, 0)]
    assert all(max(abs(dy), abs(dx)) == offset for dy, dx in neighbours)


def test_diamond_is_axial_and_square_is_diagonal():
    diamond = set(search_pattern('diamond', 2))
    square = set(search_pattern('square', 2))
    assert diamond == {(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)}
    assert square == {(0, 0), (-2, -2), (-2, 2), (2, -2), (2, 2)}


def test_unknown_shape_and_bad_offset_raise():
    with pytest.raises(ValueError):
        search_pattern('hexagon', 2)
    with pytest.raises(ValueError):
        search_pattern('diamond', 0)


def test_heuristic_choices_match_pattern_shapes():
    """main.py spells out the --heuristic choices to keep `--help` torch-free; this is
    the mechanism that stops them drifting from the real pattern builder."""
    from main import build_parser
    action = next(a for a in build_parser()._actions if a.dest == 'heuristic')
    assert sorted(action.choices) == sorted(PATTERN_SHAPES)


# ----------------------------------------------------------------------- matcher

@pytest.mark.parametrize('shape', PATTERN_SHAPES)
@pytest.mark.parametrize('pool', [1, 2])
def test_matcher_decodes_vectors_in_full_resolution_pixels(shape, pool):
    """Whatever grid the search runs on, decoded MVs are in full-resolution pixels.

    This is the contract the metrics and compensators depend on: MVC and mean_mv_mag
    are defined in source pixels and must not change units with the pooling factor.
    """
    m = PatternBlockMatcher(BS, shape, offset=4, pool=pool)
    assert m.reach_px == 4
    expected = torch.tensor(search_pattern(shape, 4, pool), dtype=torch.float32) * pool
    assert torch.equal(m.pattern_lookup, expected)
    assert m.pattern_lookup.abs().max().item() == 4.0


# --------------------------------------------------------------- pooled search

@pytest.mark.parametrize('offset,pool,expected', [
    (2, 1, 2), (2, 2, 1), (4, 2, 2), (8, 4, 2), (2, 4, 1), (1, 8, 1), (6, 4, 2),
])
def test_quantise_offset_rounds_magnitude_and_never_reaches_zero(offset, pool, expected):
    assert quantise_offset(offset, pool) == expected


@pytest.mark.parametrize('shape', SPARSE_PATTERN_SHAPES)
@pytest.mark.parametrize('pool', [2, 4])
def test_quantised_patterns_stay_symmetric(shape, pool):
    """Floor division would send -2 to -1 and +2 to 0, giving a pattern that reaches
    further one way than the other and silently biasing the motion field."""
    pattern = search_pattern(shape, 4, pool)
    assert len(pattern) == 5 and len(set(pattern)) == 5
    for dy, dx in pattern:
        assert (-dy, -dx) in pattern, f'{(dy, dx)} has no opposite in {pattern}'


def test_grid_pattern_fills_the_square():
    assert len(search_pattern('grid', 8, 4)) == 25        # r = 2 -> 5x5
    assert len(search_pattern('grid', 2, 1)) == 25        # r = 2 -> 5x5
    assert (0, 0) in search_pattern('grid', 8, 4)


def test_pooled_search_reports_vectors_on_the_pooled_grid():
    """At pool P the reachable vectors are exactly the pattern scaled by P."""
    torch.manual_seed(0)
    m = PatternBlockMatcher(BS, 'diamond', offset=4, pool=2)
    mvs, _ = m(torch.randn(2, 1, H, W), torch.randn(2, 1, H, W))
    assert set(mvs.unique().tolist()) <= {-4.0, 0.0, 4.0}


def test_pooling_the_stack_once_matches_pooling_each_slice():
    """The shared-pyramid fast path must be numerically identical to the naive one."""
    torch.manual_seed(1)
    stack = torch.randn(5, 1, H, W)
    curr, ref = stack[1:], stack[:-1]
    m = PatternBlockMatcher(BS, 'diamond', offset=4, pool=2)
    a_mv, a_sad = m(curr, ref)
    b_mv, b_sad = m(curr, ref, frame_stack=stack)
    assert torch.equal(a_mv, b_mv) and torch.equal(a_sad, b_sad)


def test_pooled_pair_is_a_noop_at_pool_one():
    c, r = torch.randn(1, 1, H, W), torch.randn(1, 1, H, W)
    pc, pr = pooled_pair(c, r, 1)
    assert pc is c and pr is r


@pytest.mark.parametrize('block_size,pool,msg', [
    (32, 5, 'does not divide'),
    (8, 4, 'at least 4x4'),
])
def test_matcher_rejects_impossible_geometry(block_size, pool, msg):
    with pytest.raises(ValueError, match=msg):
        PatternBlockMatcher(block_size, 'diamond', offset=2, pool=pool)


def test_engine_rejects_impossible_residual_geometry():
    m = PatternBlockMatcher(BS, 'diamond')
    with pytest.raises(ValueError, match='at least 8x8'):
        EVCATemporalEngine(m, {'mvc': MetricMVC()}, residual_pool=8)
    with pytest.raises(ValueError, match='does not divide'):
        EVCATemporalEngine(m, {'mvc': MetricMVC()}, residual_pool=5)


def test_engine_pools_the_residual_domain_but_not_the_vectors():
    """State frames and block size shrink with residual_pool; MVs stay in source px."""
    stack = torch.randn(3, 1, H, W)
    engine = EVCATemporalEngine(PatternBlockMatcher(BS, 'diamond', offset=4),
                                {'mvc': MetricMVC()}, residual_pool=2)
    _, state = engine(stack[1:], stack[:-1], frame_stack=stack)
    assert state.pool == 2 and state.bs == BS // 2
    assert state.current_frame.shape[-2:] == (H // 2, W // 2)
    assert state.residual.shape[-2:] == (BS // 2, BS // 2)
    assert state.mvs.abs().max().item() in (0.0, 4.0)      # full-resolution pixels


def test_forward_output_shapes():
    m = PatternBlockMatcher(BS, 'diamond')
    mvs, sad = m(torch.randn(2, 1, H, W), torch.randn(2, 1, H, W))
    assert mvs.shape == (2, 2, H // BS, W // BS)
    assert sad.shape == (2, 1, H // BS, W // BS)


def test_zero_motion_case():
    m = PatternBlockMatcher(BS, 'diamond')
    frame = torch.ones(1, 1, H, W)
    mvs, sad = m(frame, frame)
    assert torch.allclose(mvs, torch.zeros_like(mvs))
    assert torch.allclose(sad, torch.zeros_like(sad))


# ----------------------------------------------------------------- TemporalState

def test_mc_blocks_without_mvs_raises():
    state = TemporalState(torch.randn(1, 1, H, W), torch.randn(1, 1, H, W), bs=BS)
    with pytest.raises(ValueError):
        _ = state.mc_blocks


def test_lazy_mc_blocks_and_residual_are_shaped_and_cached():
    state = TemporalState(torch.randn(1, 1, H, W), torch.randn(1, 1, H, W), bs=BS)
    state.mvs = torch.zeros(1, 2, 2, 2)
    state.sad_map = torch.zeros(1, 1, 2, 2)

    mc, res = state.mc_blocks, state.residual
    assert mc.shape == (1, 1, 2, 2, BS, BS)      # [B, C, H_b, W_b, bs, bs]
    assert res.shape == (1, 1, 2, 2, BS, BS)
    assert state.mc_blocks is mc and state.residual is res


# ----------------------------------------------------------------------- metrics

def test_metric_mvc_separates_smooth_from_chaotic_fields():
    metric = MetricMVC()
    state = TemporalState(torch.zeros(1, 1, H, W), torch.zeros(1, 1, H, W))
    state.mvs = torch.zeros(1, 2, 4, 4)
    assert metric(state).item() == 0.0
    state.mvs = torch.randn(1, 2, 4, 4)
    assert metric(state).item() > 0.0


def test_metric_tcsad_averages_the_sad_map():
    state = TemporalState(torch.zeros(1, 1, H, W), torch.zeros(1, 1, H, W))
    state.sad_map = torch.full((1, 1, 2, 2), 5.0)
    assert MetricsTCSAD()(state).item() == pytest.approx(5.0)


def test_engine_pipeline():
    engine = EVCATemporalEngine(PatternBlockMatcher(BS, 'diamond'),
                                {'mvc': MetricMVC(), 'tc_sad': MetricsTCSAD()})
    results, state = engine(torch.randn(2, 1, H, W), torch.randn(2, 1, H, W))
    assert results['mvc'].shape == (2,) and results['tc_sad'].shape == (2,)
    assert state.mvs is not None and state.sad_map is not None

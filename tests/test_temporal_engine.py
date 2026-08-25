"""Motion estimator, TemporalState laziness, and the metric plugins."""
import pytest
import torch

from libs.temporal_engine import (EVCATemporalEngine, MetricMVC, MetricsTCSAD,
                                  PATTERN_SHAPES, SparsePatternBlockMatcher,
                                  TemporalState, search_pattern)

BS, H, W = 32, 64, 64


# ----------------------------------------------------------------- search pattern

@pytest.mark.parametrize('shape', PATTERN_SHAPES)
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
def test_matcher_decodes_pattern_at_full_resolution(shape):
    """MVs are read straight out of the pattern: no half-resolution rescaling."""
    m = SparsePatternBlockMatcher(BS, shape, offset=3)
    assert m.reach == 3
    assert torch.equal(m.pattern_lookup,
                       torch.tensor(search_pattern(shape, 3), dtype=torch.float32))


def test_forward_output_shapes():
    m = SparsePatternBlockMatcher(BS, 'diamond')
    mvs, sad = m(torch.randn(2, 1, H, W), torch.randn(2, 1, H, W))
    assert mvs.shape == (2, 2, H // BS, W // BS)
    assert sad.shape == (2, 1, H // BS, W // BS)


def test_zero_motion_case():
    m = SparsePatternBlockMatcher(BS, 'diamond')
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
    engine = EVCATemporalEngine(SparsePatternBlockMatcher(BS, 'diamond'),
                                {'mvc': MetricMVC(), 'tc_sad': MetricsTCSAD()})
    results, state = engine(torch.randn(2, 1, H, W), torch.randn(2, 1, H, W))
    assert results['mvc'].shape == (2,) and results['tc_sad'].shape == (2,)
    assert state.mvs is not None and state.sad_map is not None

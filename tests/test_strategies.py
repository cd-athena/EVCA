"""Phase 2 strategy plumbing: DCT backends, MC variants, presets, gating."""
import numpy as np
import pytest
import torch

from libs.motion_compensation import (BlockMC, DenseMC, OBMC, build_compensator,
                                      obmc_weights, vector_median)
from libs.transforms import dct_2d_matmul, dct_2d_torchdct
from libs.weight_dct import weight_dct
from main import PRESETS, _add_arguments, get_parser_arguments
from tests.conftest import make_args, run_evca, write_raw_yuv
from validation.synthetic import gen_translation

torch_dct = pytest.importorskip('torch_dct', reason='optional ablation dependency')


# --------------------------------------------------------------------------- DCT

@pytest.mark.parametrize('n', [16, 32])
def test_dct_impl_parity(n):
    """matmul and torch_dct must agree to 1e-4 relative on random blocks."""
    torch.manual_seed(0)
    blocks = torch.randn(64, n, n) * 60.0
    a, b = dct_2d_matmul(blocks), dct_2d_torchdct(blocks)
    rel = (a - b).abs().max() / b.abs().max()
    assert rel < 1e-4, f'relative disagreement {rel:.2e}'


def test_dct_impl_parity_end_to_end(tmp_path):
    """Selecting either backend must not move the reported metrics."""
    frames, _ = gen_translation(96, 128, 6, vy=0, vx=2, seed=41)
    path = tmp_path / 'dct.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8)
    out = {}
    for impl in ('matmul', 'torch_dct'):
        args = make_args(input=str(path), resolution='128x96', dct_impl=impl,
                         csv=str(tmp_path / f'{impl}.csv'),
                         motion_estimation=True, profile='full')
        out[impl] = run_evca(args)
    for col in out['matmul'].columns:
        assert np.allclose(out['matmul'][col], out['torch_dct'][col],
                           rtol=1e-4, atol=1e-4), col


# ---------------------------------------------------------------------- weights

def test_residual_dc_weight():
    args = make_args(block_size=32)
    dev = torch.device('cpu')
    assert weight_dct(args, dev)[0, 0].item() == 0.0
    kept = weight_dct(args, dev, keep_dc=True)[0, 0].item()
    assert kept == pytest.approx(np.exp((1 / 1024) ** 2 - 1.0), rel=1e-6)
    # every other coefficient is untouched
    assert torch.equal(weight_dct(args, dev)[1:], weight_dct(args, dev, keep_dc=True)[1:])


# --------------------------------------------------------------------------- MC

def _frames(seed=51, shift=4):
    f, _ = gen_translation(128, 160, 3, vy=0, vx=shift, seed=seed)
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).unsqueeze(0).float()
    return to_t(f[2]), to_t(f[1])


@pytest.mark.parametrize('mc,smooth', [('dense_smooth', 'gauss'), ('dense_smooth', 'median'),
                                       ('dense_smooth', 'none'), ('dense', 'gauss'),
                                       ('block', 'gauss'), ('obmc', 'gauss')])
def test_compensators_reduce_residual_on_global_pan(mc, smooth):
    """Every MC variant must beat no compensation on a uniform in-range translation."""
    curr, ref = _frames()
    mvs = torch.zeros(1, 2, 4, 5)
    mvs[:, 1] = 4.0                                    # dx = +4 everywhere
    comp = build_compensator(mc, smooth)
    pred = comp(ref, mvs, 32)
    interior = (slice(None), slice(None), slice(32, -32), slice(32, -32))
    mc_err = (curr[interior] - pred[interior]).abs().mean()
    raw_err = (curr[interior] - ref[interior]).abs().mean()
    assert mc_err < 0.05 * raw_err, f'{mc}/{smooth}: {mc_err:.3f} vs raw {raw_err:.3f}'


def test_dense_and_dense_smooth_agree_on_uniform_field():
    """Smoothing a constant MV field is a no-op, so the two dense modes coincide."""
    curr, ref = _frames()
    mvs = torch.full((1, 2, 4, 5), 2.0)
    a = build_compensator('dense_smooth', 'gauss')(ref, mvs, 32)
    b = build_compensator('dense', 'gauss')(ref, mvs, 32)
    assert torch.allclose(a, b, atol=1e-4)


def test_obmc_weights_are_partition_of_unity():
    w = obmc_weights(32, torch.device('cpu'))
    assert w.shape == (5, 1, 32, 32)
    assert torch.allclose(w.sum(dim=0), torch.ones(1, 32, 32), atol=1e-6)
    assert (w >= 0).all()


def test_obmc_preserves_dc():
    """A partition-of-unity blend must not change a flat frame's level."""
    ref = torch.full((1, 1, 128, 160), 100.0)
    mvs = torch.zeros(1, 2, 4, 5)
    out = OBMC()(ref, mvs, 32)
    assert torch.allclose(out, ref, atol=1e-3)


def test_vector_median_picks_an_existing_vector():
    """The vector median must return one of the input vectors, never a new one."""
    torch.manual_seed(3)
    mvs = torch.randint(-6, 7, (1, 2, 5, 5)).float()
    out = vector_median(mvs)
    padded = torch.nn.functional.pad(mvs, (1, 1, 1, 1), mode='replicate')
    for y in range(5):
        for x in range(5):
            neigh = padded[0, :, y:y + 3, x:x + 3].reshape(2, 9).T
            assert (neigh == out[0, :, y, x]).all(dim=1).any()


def test_vector_median_rejects_outlier():
    """A lone outlier surrounded by agreement is replaced by the consensus vector."""
    mvs = torch.zeros(1, 2, 3, 3)
    mvs[:, 1] = 2.0
    mvs[0, :, 1, 1] = torch.tensor([9.0, -9.0])
    out = vector_median(mvs)
    assert out[0, 0, 1, 1].item() == 0.0 and out[0, 1, 1, 1].item() == 2.0


# ------------------------------------------------------------------------ flags

def test_preset_iter4_pins_the_reference_search_pattern():
    """`--preset iter4` must reproduce the Iteration-4 reference, not today's defaults.

    Phase 3 changed the default search pattern from the axis-only 13-point plus to the
    17-point diagonal-carrying `diamond`, so the preset and the defaults now diverge on
    exactly that axis and agree everywhere else. This is the preset earning its keep:
    before this change the two were identical and the pin was vacuous.
    """
    defaults = get_parser_arguments([])
    preset = get_parser_arguments(['--preset', 'iter4'])
    assert preset.heuristic == 'diamond_axis'
    assert defaults.heuristic == 'diamond'
    for dest, value in PRESETS['iter4'].items():
        assert getattr(preset, dest) == value, dest
        if dest != 'heuristic':
            assert getattr(defaults, dest) == value, dest


def test_heuristic_choices_match_patterns():
    """main.py spells out the --heuristic choices to keep `--help` torch-free; this is
    the mechanism that stops them drifting from the real pattern table."""
    import argparse

    from libs.temporal_engine import SEARCH_PATTERNS
    parser = argparse.ArgumentParser(add_help=False)
    _add_arguments(parser)
    action = next(a for a in parser._actions if a.dest == 'heuristic')
    assert sorted(action.choices) == sorted(SEARCH_PATTERNS)


@pytest.mark.parametrize('name', sorted(['diamond', 'diamond_axis', 'diamond_dense', 'square']))
def test_every_pattern_builds_and_decodes(name):
    """Each pattern must be unique, centred, and decode to even full-res vectors."""
    from libs.temporal_engine import SEARCH_PATTERNS, SparsePatternBlockMatcher
    pattern = SEARCH_PATTERNS[name]
    assert len(set(pattern)) == len(pattern), 'duplicate candidate'
    assert (0, 0) in pattern, 'pattern must be able to report zero motion'
    m = SparsePatternBlockMatcher(block_size=32, heuristic=name)
    assert m.max_reach_fullres == 2 * max(max(abs(a), abs(b)) for a, b in pattern)
    assert torch.equal(m.pattern_lookup, torch.tensor(pattern, dtype=torch.float32) * 2.0)


def test_diamond_patterns_represent_diagonal_motion():
    """The default pattern must be able to express diagonal motion.

    `diamond_axis` cannot -- every candidate lies on an axis -- which is why a true
    (4, 4) translation used to be estimated as magnitude 4.00 against a true 5.66.
    """
    from libs.temporal_engine import SEARCH_PATTERNS
    diagonals = lambda p: [c for c in SEARCH_PATTERNS[p] if c[0] != 0 and c[1] != 0]
    assert diagonals('diamond_axis') == [], 'reference pattern is axis-only by definition'
    assert len(diagonals('diamond')) == 4
    assert len(diagonals('diamond_dense')) == 8
    # (+/-4, +/-4) full-res must be reachable in the default pattern
    assert (2, 2) in SEARCH_PATTERNS['diamond']


def test_default_pattern_beats_reference_on_diagonal_motion():
    """End-to-end: the new default must estimate a 45-degree pan that the old one missed."""
    from libs.temporal_engine import SparsePatternBlockMatcher
    frames, _ = gen_translation(256, 320, 3, vy=4, vx=4, seed=11)
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(np.rint(a))).float()[None, None]
    curr, ref = to_t(frames[2]), to_t(frames[1])
    err = {}
    for name in ('diamond_axis', 'diamond'):
        mvs, _ = SparsePatternBlockMatcher(block_size=32, heuristic=name)(curr, ref)
        interior = mvs[0, :, 1:-1, 1:-1]
        err[name] = torch.sqrt((interior[0] - 4) ** 2 + (interior[1] - 4) ** 2).mean().item()
    assert err['diamond'] < 0.01, f"default pattern should be exact, got {err['diamond']:.3f}"
    assert err['diamond_axis'] > 4.0, 'reference pattern is expected to fail here'


def test_preset_yields_to_explicit_flag():
    args = get_parser_arguments(['--preset', 'iter4', '--gate', 'none'])
    assert args.gate == 'none' and args.mc == 'dense_smooth'


def test_unimplemented_me_flags_raise(tmp_path):
    """Phase-3 flags must fail loudly rather than silently running the default search."""
    from libs.EVCA import build_motion_estimator
    for override in [{'me': 'hierarchical'}, {'me_subpel': 1}, {'me_predictor': 'global'},
                     {'me_lambda': 1.5}, {'me_merge': True}, {'me_criterion': 'satd'}]:
        with pytest.raises(NotImplementedError):
            build_motion_estimator(make_args(**override), 1920)


def test_gate_none_allows_residual_above_sc(tmp_path):
    """With gating off, TC_MC may exceed SC; with gating on it never can."""
    frames, _ = gen_translation(96, 128, 6, vy=0, vx=32, seed=61)   # far out of range
    path = tmp_path / 'gate.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8)
    res = {}
    for gate in ('intra', 'none'):
        args = make_args(input=str(path), resolution='128x96', gate=gate,
                         csv=str(tmp_path / f'{gate}.csv'),
                         motion_estimation=True, profile='full')
        res[gate] = run_evca(args)
    assert (res['intra'].loc[1:, 'TC_MC'] <= res['intra'].loc[1:, 'SC'] + 1e-4).all()
    assert (res['none'].loc[1:, 'TC_MC'] >= res['intra'].loc[1:, 'TC_MC'] - 1e-4).all()
    assert (res['none'].loc[1:, 'TC_MC'] > res['none'].loc[1:, 'SC']).any()


def test_residual_dc_increases_tcmc(tmp_path):
    """Keeping DC can only add energy to the residual transform."""
    frames, _ = gen_translation(96, 128, 6, vy=0, vx=16, seed=62)
    path = tmp_path / 'dc.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8)
    res = {}
    for dc in (False, True):
        args = make_args(input=str(path), resolution='128x96', residual_dc=dc,
                         gate='none', csv=str(tmp_path / f'dc{dc}.csv'),
                         motion_estimation=True, profile='full')
        res[dc] = run_evca(args)
    assert (res[True].loc[1:, 'TC_MC'] >= res[False].loc[1:, 'TC_MC'] - 1e-6).all()

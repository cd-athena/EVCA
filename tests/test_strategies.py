"""Phase 2 strategy plumbing: DCT backends, MC variants, presets, gating."""
import numpy as np
import pytest
import torch

from libs.motion_compensation import (BlockMC, DenseMC, OBMC, build_compensator,
                                      obmc_weights, vector_median)
from libs.transforms import dct_2d_matmul, dct_2d_torchdct
from libs.weight_dct import weight_dct
from main import PRESETS, get_parser_arguments
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

def test_preset_iter4_matches_defaults():
    """`--preset iter4` must reproduce today's defaults exactly (Phase 4 contract)."""
    defaults = get_parser_arguments([])
    for dest, value in PRESETS['iter4'].items():
        assert getattr(defaults, dest) == value, dest


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

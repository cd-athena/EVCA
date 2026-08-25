"""Motion-compensation variants, MV smoothing, and the residual/gate flags."""
import numpy as np
import pytest
import torch

from libs.motion_compensation import (OBMC, build_compensator, obmc_weights,
                                      vector_median)
from libs.weight_dct import weight_dct
from tests.conftest import make_args, run_evca, write_raw_yuv
from validation.synthetic import gen_static_noise, gen_translation


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
    pred = build_compensator(mc, smooth)(ref, mvs, 32)
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
    out = OBMC()(ref, torch.zeros(1, 2, 4, 5), 32)
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


# ------------------------------------------------------------------- gate / DC

def test_gate_caps_tcmc_at_sc(tmp_path):
    """The intra gate is unconditional: TC_MC can never exceed the frame's own SC.

    Driven with motion far outside the search pattern, so the uncompensated residual
    would otherwise carry more energy than coding the block intra.
    """
    frames, _ = gen_translation(96, 128, 6, vy=0, vx=32, seed=61)
    path = tmp_path / 'gate.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8)
    df = run_evca(make_args(input=str(path), resolution='128x96',
                            csv=str(tmp_path / 'gate.csv'),
                            motion_estimation=True, profile='full'))
    assert (df.loc[1:, 'TC_MC'] <= df.loc[1:, 'SC'] + 1e-4).all()
    # With the search this badly outmatched the gate must actually be binding.
    assert (df.loc[1:, 'intra_frac'] > 0).all()


def test_residual_dc_increases_tcmc(tmp_path):
    """Keeping DC can only add energy to the residual transform.

    Driven with static content plus per-frame noise: the search matches at (0, 0), so
    the residual is real but far below intra energy and the gate does not bind, leaving
    the DC term visible in TC_MC.
    """
    frames, _ = gen_static_noise(96, 128, 6, sigma=6.0, seed=62)
    path = tmp_path / 'dc.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8)
    res = {}
    for dc in (False, True):
        res[dc] = run_evca(make_args(input=str(path), resolution='128x96',
                                     residual_dc=dc, csv=str(tmp_path / f'dc{dc}.csv'),
                                     motion_estimation=True, profile='full'))
    assert (res[True].loc[1:, 'TC_MC'] >= res[False].loc[1:, 'TC_MC'] - 1e-6).all()
    # Not vacuous: the gate must be leaving TC_MC below SC so the DC term can show.
    assert (res[True].loc[1:, 'intra_frac'] == 0).all()
    assert (res[True].loc[1:, 'TC_MC'] > res[False].loc[1:, 'TC_MC']).any()

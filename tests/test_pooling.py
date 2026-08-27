"""Downsampled temporal path: pool 1 is a no-op, and the pooled path stays coherent.

The load-bearing test here is `test_pool_one_reproduces_the_unpooled_pipeline`: the
whole feature is only landable because the default reproduces the previous behaviour
exactly, so that regression is pinned first and everything else builds on it.
"""
import numpy as np
import pytest
import torch

from libs.EVCA import resolve_pools
from tests.conftest import make_args, run_evca, write_raw_yuv
from validation.synthetic import gen_translation, gen_static_noise

RES = '256x192'          # 8 x 6 blocks of 32, divisible by 4 so pool 4 stays aligned
W, H = 256, 192


@pytest.fixture(scope='module')
def pan_yuv(tmp_path_factory):
    """A 6 px/frame pan: outside a +/-2 px pattern, inside a pool-4 one."""
    d = tmp_path_factory.mktemp('pool')
    frames, _ = gen_translation(H, W, 8, vy=0, vx=6, seed=71)
    path = d / 'pan.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8)
    return str(path), str(d)


def _run(path, out_dir, tag, **overrides):
    return run_evca(make_args(input=path, resolution=RES, csv=f'{out_dir}/{tag}.csv',
                              motion_estimation=True, profile='full', **overrides))


# --------------------------------------------------------------- pool resolution

@pytest.mark.parametrize('temporal_pool,me_pool,expected', [
    (1, None, (1, 1)),
    (2, None, (2, 2)),       # one flag pools the whole path
    (1, 2, (2, 1)),          # search pooled, residual not
    (2, 1, (1, 2)),          # residual pooled, search not
])
def test_me_pool_defaults_to_temporal_pool(temporal_pool, me_pool, expected):
    args = make_args(temporal_pool=temporal_pool, me_pool=me_pool)
    assert resolve_pools(args) == expected


# ------------------------------------------------------------------- the no-op

def test_pool_one_reproduces_the_unpooled_pipeline(pan_yuv):
    """Explicit pool 1 must be indistinguishable from not passing the flags at all."""
    path, d = pan_yuv
    default = _run(path, d, 'default')
    explicit = _run(path, d, 'explicit', temporal_pool=1, me_pool=1)
    for col in default.columns:
        assert np.array_equal(default[col].to_numpy(), explicit[col].to_numpy()), col


@pytest.mark.parametrize('temporal_pool', [2, 4])
def test_pooling_leaves_the_spatial_metrics_untouched(pan_yuv, temporal_pool):
    """Pooling is confined to the temporal path: SC and friends are still computed at
    full resolution and must not move, since they are separately validated metrics."""
    path, d = pan_yuv
    base = _run(path, d, 'sbase')
    pooled = _run(path, d, f'spool{temporal_pool}', temporal_pool=temporal_pool)
    for col in ('B', 'SC', 'TC', 'TC2'):
        assert np.array_equal(base[col].to_numpy(), pooled[col].to_numpy()), col


# ------------------------------------------------------------ pooled invariants

@pytest.mark.parametrize('temporal_pool', [1, 2])
def test_intra_gate_still_caps_tcmc_at_sc(pan_yuv, temporal_pool):
    """The gate is unconditional at every pooling factor.

    At pool > 1 both sides are recomputed on pooled blocks so they share a scale; if
    that pairing were wrong the cap would compare incommensurate energies and this
    would fail.
    """
    path, d = pan_yuv
    df = _run(path, d, f'gate{temporal_pool}', temporal_pool=temporal_pool)
    assert (df.loc[1:, 'TC_MC'] <= df.loc[1:, 'SC'] + 1e-4).all()
    assert (df.loc[1:, 'TC_MC'] >= 0).all()


def test_wider_pooled_search_lowers_the_residual_on_an_out_of_reach_pan(pan_yuv):
    """A 6 px pan is unreachable at +/-2 px; a pool-2 search with +/-6 px reach finds
    it, and the motion-compensated residual falls as a result."""
    path, d = pan_yuv
    narrow = _run(path, d, 'narrow')
    wide = _run(path, d, 'wide', temporal_pool=2, me_offset=6)
    assert wide.loc[1:, 'mean_mv_mag'].mean() > narrow.loc[1:, 'mean_mv_mag'].mean()
    assert wide.loc[1:, 'intra_frac'].mean() < narrow.loc[1:, 'intra_frac'].mean()


def test_static_content_reports_zero_motion_at_every_pool(tmp_path):
    """Pooling must not invent motion where there is none."""
    frames, _ = gen_static_noise(H, W, 5, sigma=2.0, seed=72)
    path = tmp_path / 'static.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8)
    for pool in (1, 2, 4):
        df = _run(str(path), str(tmp_path), f'static{pool}', temporal_pool=pool)
        assert df.loc[1:, 'mean_mv_mag'].max() == 0.0, f'pool {pool} invented motion'


# ------------------------------------------------------------------ the guards

def test_transform_guard_rejects_pooled_dct_b(pan_yuv):
    """DCT_B is hard-wired to a 32x32 block, so it cannot serve a pooled residual."""
    path, d = pan_yuv
    with pytest.raises(ValueError, match='requires --transform DCT'):
        _run(path, d, 'dctb', temporal_pool=2, transform='DCT_B')


def test_block_size_guard_rejects_a_degenerate_residual_block(pan_yuv):
    path, d = pan_yuv
    with pytest.raises(ValueError, match='at least 8x8'):
        _run(path, d, 'tiny', temporal_pool=4, block_size=16)

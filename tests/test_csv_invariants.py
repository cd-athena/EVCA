"""Frame-level CSV invariants per profile (Phase 1.4): row counts, frame-0 padding,
column sets, and 8/10-bit scaling parity."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_args, run_evca, write_raw_yuv
from validation.synthetic import gen_translation

BASE_COLS = ['B', 'SC', 'TC', 'TC2']
FAST_ME_COLS = BASE_COLS + ['MVC', 'TC_SAD', 'MV_sat_frac', 'mean_mv_mag']
FULL_ME_COLS = BASE_COLS + ['MVC', 'TC_SAD', 'TC_MC', 'MV_sat_frac', 'mean_mv_mag', 'intra_frac']

N_FRAMES, HEIGHT, WIDTH = 10, 96, 128


@pytest.fixture(scope='module')
def yuv8(tmp_path_factory):
    d = tmp_path_factory.mktemp('csv_inv')
    frames, _ = gen_translation(HEIGHT, WIDTH, N_FRAMES, vy=0, vx=2, seed=21)
    path = d / 'inv8.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8, chroma_seed=3)
    return str(path)


def _run(yuv, tmp_path, name, **overrides):
    args = make_args(input=yuv, resolution=f'{WIDTH}x{HEIGHT}',
                     csv=str(tmp_path / f'{name}.csv'), **overrides)
    return args, run_evca(args)


def test_baseline_columns_and_rows(yuv8, tmp_path):
    args, df = _run(yuv8, tmp_path, 'base')
    assert list(df.columns) == BASE_COLS
    assert len(df) == N_FRAMES
    assert df.loc[0, 'TC'] == 0.0 and df.loc[0, 'TC2'] == 0.0 and df.loc[1, 'TC2'] == 0.0
    assert (df.loc[1:, 'TC'] > 0).all()


def test_fast_profile_columns(yuv8, tmp_path):
    args, df = _run(yuv8, tmp_path, 'fast', motion_estimation=True, profile='fast')
    assert list(df.columns) == FAST_ME_COLS
    assert len(df) == N_FRAMES
    for col in ['MVC', 'TC_SAD', 'MV_sat_frac', 'mean_mv_mag']:
        assert df.loc[0, col] == 0.0, f'frame-0 padding missing for {col}'
    # constant vx=2 translation: mean MV magnitude ~2 on frames >= 1
    assert abs(df.loc[2, 'mean_mv_mag'] - 2.0) < 0.5


def test_full_profile_columns(yuv8, tmp_path):
    args, df = _run(yuv8, tmp_path, 'full', motion_estimation=True, profile='full')
    assert list(df.columns) == FULL_ME_COLS
    assert len(df) == N_FRAMES
    assert df.loc[0, 'TC_MC'] == 0.0 and df.loc[0, 'intra_frac'] == 0.0
    # gated: TC_MC can never exceed SC of the same frame
    assert (df.loc[1:, 'TC_MC'] <= df.loc[1:, 'SC'] + 1e-4).all()


def test_sample_rate_row_count(yuv8, tmp_path):
    args, df = _run(yuv8, tmp_path, 'sr3', sample_rate=3, motion_estimation=True)
    assert len(df) == len(range(0, N_FRAMES, 3))


def test_gop_smaller_than_file(yuv8, tmp_path):
    """Multiple GOPs per file must still yield one row per sampled frame."""
    args, df = _run(yuv8, tmp_path, 'gop4', gopsize=4, motion_estimation=True, profile='full')
    assert len(df) == N_FRAMES
    assert (df.loc[1:, 'TC_SAD'] > 0).all()


def test_provenance_sidecar(yuv8, tmp_path):
    args, _ = _run(yuv8, tmp_path, 'prov')
    meta_path = args.csv + '.meta.json'
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    assert 'git_sha' in meta and 'args' in meta
    assert meta['args']['resolution'] == f'{WIDTH}x{HEIGHT}'


def test_bitdepth_scaling_parity(tmp_path):
    """Identical content at 8 and 10 bit must produce matching scaled metrics."""
    frames, _ = gen_translation(HEIGHT, WIDTH, 6, vy=0, vx=2, seed=22)
    y8 = [np.rint(f).astype(np.int64) for f in frames]
    y10 = [y * 4 for y in y8]
    p8, p10 = tmp_path / 'c8.yuv', tmp_path / 'c10.yuv'
    write_raw_yuv(p8, y8, bit_depth=8)
    write_raw_yuv(p10, y10, bit_depth=10)

    args8 = make_args(input=str(p8), resolution=f'{WIDTH}x{HEIGHT}',
                      csv=str(tmp_path / 'c8.csv'), motion_estimation=True, profile='full')
    args10 = make_args(input=str(p10), resolution=f'{WIDTH}x{HEIGHT}', bit_depth=10,
                       csv=str(tmp_path / 'c10.csv'), motion_estimation=True, profile='full')
    df8, df10 = run_evca(args8), run_evca(args10)

    for col in ['SC', 'TC', 'TC2', 'TC_SAD', 'TC_MC']:
        assert np.allclose(df8[col], df10[col], rtol=1e-4, atol=1e-4), col
    # MVC is measured in pixels and must be bit-depth invariant (identical MVs)
    assert np.allclose(df8['MVC'], df10['MVC'], rtol=1e-5, atol=1e-6)

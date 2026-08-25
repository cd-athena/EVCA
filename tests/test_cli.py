"""End-to-end CLI smoke tests: each method reaches a CSV with its own columns.

Deliberately thin. These exist to catch import errors, argument-plumbing breakage and
crashes in the subprocess path; the numeric behaviour of the pipeline is covered
in-process (and far faster) by test_csv_invariants.py.
"""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tests.conftest import write_raw_yuv
from validation.synthetic import gen_translation

N_FRAMES, HEIGHT, WIDTH = 6, 96, 128


@pytest.fixture(scope='module')
def yuv(tmp_path_factory):
    frames, _ = gen_translation(HEIGHT, WIDTH, N_FRAMES, vy=0, vx=2, seed=71)
    path = tmp_path_factory.mktemp('cli') / 'cli.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8, chroma_seed=5)
    return str(path)


def run_cli(yuv, csv, *extra):
    cmd = [sys.executable, 'main.py', '-i', yuv, '-r', f'{WIDTH}x{HEIGHT}',
           '--device', 'cpu', '-c', str(csv), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f'{" ".join(extra)} failed:\n{proc.stderr}'
    return pd.read_csv(csv)


def test_help_is_generated_and_needs_no_torch():
    """`--help` must render from the parser without loading torch.
    """
    out = subprocess.run([sys.executable, 'main.py', '--help'],
                         capture_output=True, text=True, check=True).stdout
    for expected in ('--me-offset', '--heuristic', 'motion estimation', 'examples:'):
        assert expected in out, expected

    trace = subprocess.run([sys.executable, '-X', 'importtime', 'main.py', '--help'],
                           capture_output=True, text=True, check=True).stderr
    imported = {line.rsplit('|', 1)[-1].strip() for line in trace.splitlines()}
    assert not {m for m in imported if m == 'torch' or m.startswith('torch.')}, \
        'main.py imported torch just to print --help'


def test_evca_default(yuv, tmp_path):
    df = run_cli(yuv, tmp_path / 'evca.csv')
    assert list(df.columns) == ['B', 'SC', 'TC', 'TC2']
    assert len(df) == N_FRAMES


def test_vca_method(yuv, tmp_path):
    df = run_cli(yuv, tmp_path / 'vca.csv', '-m', 'VCA')
    assert 'h' in df.columns


def test_siti_method(yuv, tmp_path):
    df = run_cli(yuv, tmp_path / 'siti.csv', '-m', 'SITI')
    assert list(df.columns) == ['SI', 'TI', 'TI2']
    assert len(df) == N_FRAMES


def test_motion_estimation_full_profile(yuv, tmp_path):
    df = run_cli(yuv, tmp_path / 'me.csv', '-me', '-cc', '-cf', '--profile', 'full')
    for col in ['MVC', 'TC_SAD', 'TC_MC', 'mean_mv_mag', 'intra_frac',
                'SC_u', 'SC_v', 'Colorfulness']:
        assert col in df.columns, col

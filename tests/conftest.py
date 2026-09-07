"""Shared fixtures/helpers for the EVCA test suite (CPU-only, fast)."""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def make_args(**overrides):
    """Argument namespace with the real CLI defaults (stays in sync with main.py)."""
    from main import get_parser_arguments
    args = get_parser_arguments([])
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise AttributeError(f'unknown CLI arg override: {k}')
        setattr(args, k, v)
    return args


def geometry(args):
    """Replicates EVCA()'s derived geometry for direct loader calls."""
    width, height = map(int, args.resolution.split('x'))
    bytes_per_sample = 1 if args.bit_depth == 8 else 2
    if args.pix_fmt == 'yuv420':
        pix_size = 1.5 * bytes_per_sample
        uv_w, uv_h = width // 2, height // 2
        cb_size = args.block_size // 2
    else:
        pix_size = 3.0 * bytes_per_sample
        uv_w, uv_h = width, height
        cb_size = args.block_size
    luma_size = width * height
    chroma_size = uv_w * uv_h
    return width, height, pix_size, luma_size, chroma_size, uv_w, uv_h, cb_size


def write_raw_yuv(path, y_frames, bit_depth=8, chroma_seed=None):
    """Writes yuv420 frames; y_frames is a list of integer arrays already in the
    target bit-depth range. Chroma is random (seeded) or constant mid-gray."""
    height, width = y_frames[0].shape
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    mid = 128 if bit_depth == 8 else 128 * (1 << (bit_depth - 8))
    rng = np.random.default_rng(chroma_seed) if chroma_seed is not None else None
    with open(path, 'wb') as f:
        for y in y_frames:
            f.write(y.astype(dtype).tobytes())
            for _ in range(2):
                if rng is not None:
                    uv = rng.integers(0, (1 << bit_depth), size=(height // 2, width // 2))
                else:
                    uv = np.full((height // 2, width // 2), mid)
                f.write(uv.astype(dtype).tobytes())


def run_evca(args):
    """Runs the EVCA pipeline in-process on CPU and returns the frame-level DataFrame."""
    import pandas as pd
    import torch
    from libs.EVCA import EVCA
    EVCA(args, [args.input], torch.device('cpu'))
    return pd.read_csv(args.csv)


@pytest.fixture(scope='session')
def small_translation_yuv(tmp_path_factory):
    """160x128, 8 frames, integer translation vx=4 (in diamond search range)."""
    from validation.synthetic import gen_translation
    d = tmp_path_factory.mktemp('synth')
    frames, gt = gen_translation(128, 160, 8, vy=0, vx=4, seed=3)
    path = d / 'trans4.yuv'
    write_raw_yuv(path, [np.rint(f) for f in frames], bit_depth=8)
    return str(path), gt

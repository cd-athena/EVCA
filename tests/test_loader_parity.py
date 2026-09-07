"""Bit-exact tensor parity between load_gop and load_gop_optimized (Phase 0.2)."""
import numpy as np
import pytest
import torch

from libs.video_loader import load_gop, load_gop_optimized
from tests.conftest import make_args, geometry, write_raw_yuv

DEVICE = torch.device('cpu')


def _load_both(args, path, n_frames):
    width, height, pix_size, luma_size, chroma_size, uv_w, uv_h, cb_size = geometry(args)
    args.input = str(path)
    out = []
    for fn in (load_gop, load_gop_optimized):
        with open(path, 'rb') as stream:
            out.append(fn(args, stream, 0, n_frames, DEVICE,
                          width, height, pix_size, luma_size, chroma_size,
                          uv_w, uv_h, cb_size))
    return out


@pytest.mark.parametrize('bit_depth', [8, 10])
@pytest.mark.parametrize('with_chroma', [False, True])
def test_loader_parity(tmp_path, bit_depth, with_chroma):
    n_frames, height, width = 5, 96, 128
    rng = np.random.default_rng(42)
    maxval = 1 << bit_depth
    y_frames = [rng.integers(0, maxval, size=(height, width)) for _ in range(n_frames)]
    path = tmp_path / f'parity_{bit_depth}.yuv'
    write_raw_yuv(path, y_frames, bit_depth=bit_depth, chroma_seed=7)

    args = make_args(resolution=f'{width}x{height}', bit_depth=bit_depth,
                     chroma_complexity=with_chroma, colorfulness=with_chroma)
    (std, opt) = _load_both(args, path, n_frames)

    for i, name in enumerate(['Y_blocks', 'U_blocks', 'V_blocks']):
        if std[i] is None:
            assert opt[i] is None, name
        else:
            assert torch.equal(std[i], opt[i]), f'{name} differs ({bit_depth}-bit)'
    # colorfulness floats
    assert np.allclose(std[3], opt[3]), 'colorfulness differs'
    assert torch.equal(std[4], opt[4]), f'Y_frames differ ({bit_depth}-bit)'


def test_loader_parity_second_gop(tmp_path):
    """Parity must hold for a GOP that does not start at frame 0 (seek offsets)."""
    n_frames, height, width = 6, 64, 64
    rng = np.random.default_rng(1)
    y_frames = [rng.integers(0, 256, size=(height, width)) for _ in range(n_frames)]
    path = tmp_path / 'seek.yuv'
    write_raw_yuv(path, y_frames, bit_depth=8, chroma_seed=2)

    args = make_args(resolution=f'{width}x{height}', chroma_complexity=True)
    width_, height_, pix_size, luma_size, chroma_size, uv_w, uv_h, cb_size = geometry(args)
    args.input = str(path)
    outs = []
    for fn in (load_gop, load_gop_optimized):
        with open(path, 'rb') as stream:
            outs.append(fn(args, stream, 3, 6, torch.device('cpu'),
                           width_, height_, pix_size, luma_size, chroma_size,
                           uv_w, uv_h, cb_size))
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][4], outs[1][4])

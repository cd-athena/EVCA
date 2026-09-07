"""Synthetic 8-bit yuv420 sequences with known ground-truth motion, for the tests.

`write_yuv420` emits a `.yuv` (Y plane textured, U/V constant 128) plus a `.json`
sidecar holding the exact motion parameters; the generators return the frames and that
ground truth directly, which is how the tests consume them.

MV sign convention (matches SparsePatternBlockMatcher): the estimated MV (dy, dx)
of a block satisfies curr(y, x) ~= ref(y + dy, x + dx). For a sequence produced
by sliding a crop window over a fixed canvas with offset increment (vy, vx) per
frame, the expected MV is exactly (vy, vx).
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def make_base_texture(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Band-pass filtered noise plus a few hard edges.

    The band-pass component gives SAD a unique minimum under translation, and the
    injected rectangles/lines add strong edges so DCT-based metrics are non-trivial.
    Returns a float32 array in [16, 235].
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((height, width))
    bandpass = gaussian_filter(noise, sigma=1.5) - gaussian_filter(noise, sigma=6.0)
    bandpass /= np.abs(bandpass).max() + 1e-9
    img = 128.0 + 70.0 * bandpass

    # A few rectangles with hard edges
    for _ in range(8):
        h = int(rng.integers(height // 16, height // 4))
        w = int(rng.integers(width // 16, width // 4))
        y = int(rng.integers(0, height - h))
        x = int(rng.integers(0, width - w))
        img[y:y + h, x:x + w] += float(rng.choice([-45.0, 45.0]))

    # A couple of 3px-wide diagonal lines
    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(3):
        slope = float(rng.uniform(-1.5, 1.5))
        icpt = float(rng.uniform(0, height))
        mask = np.abs(yy - (slope * xx + icpt)) < 1.5
        img[mask] += float(rng.choice([-50.0, 50.0]))

    return np.clip(img, 16.0, 235.0).astype(np.float32)


def write_yuv420(path: Path, y_frames: list, gt: dict) -> None:
    """Writes 8-bit yuv420p with constant chroma and a ground-truth JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = y_frames[0].shape
    uv = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    with open(path, 'wb') as f:
        for y in y_frames:
            f.write(np.clip(np.rint(y), 0, 255).astype(np.uint8).tobytes())
            f.write(uv.tobytes())
            f.write(uv.tobytes())
    gt = dict(gt, width=width, height=height, frames=len(y_frames),
              pix_fmt='yuv420', bit_depth=8)
    with open(path.with_suffix('.json'), 'w') as f:
        json.dump(gt, f, indent=2)


def gen_translation(height: int, width: int, n_frames: int, vy: int, vx: int,
                    seed: int = 0) -> tuple:
    """Integer translation: crop window slides (vy, vx) px/frame over a fixed canvas."""
    span_y, span_x = abs(vy) * (n_frames - 1), abs(vx) * (n_frames - 1)
    canvas = make_base_texture(height + span_y + 8, width + span_x + 8, seed)
    oy0 = 4 + (span_y if vy < 0 else 0)
    ox0 = 4 + (span_x if vx < 0 else 0)
    frames = []
    for f in range(n_frames):
        oy, ox = oy0 + f * vy, ox0 + f * vx
        frames.append(canvas[oy:oy + height, ox:ox + width])
    gt = {'type': 'translation', 'mv_dy': vy, 'mv_dx': vx}
    return frames, gt


def gen_static_noise(height: int, width: int, n_frames: int, sigma: float = 4.0,
                     seed: int = 0) -> tuple:
    """Static texture plus i.i.d. Gaussian noise per frame (zero true motion)."""
    base = make_base_texture(height, width, seed)
    rng = np.random.default_rng(seed + 1)
    frames = [np.clip(base + rng.standard_normal(base.shape).astype(np.float32) * sigma,
                      0, 255) for _ in range(n_frames)]
    gt = {'type': 'static_noise', 'sigma': sigma, 'mv_dy': 0, 'mv_dx': 0}
    return frames, gt


def gen_cut(height: int, width: int, n_frames: int, seed: int = 0) -> tuple:
    """Hard cut: static texture A, then an unrelated static texture B at n_frames//2."""
    a = make_base_texture(height, width, seed)
    b = make_base_texture(height, width, seed + 100)
    cut_at = n_frames // 2
    frames = [a] * cut_at + [b] * (n_frames - cut_at)
    gt = {'type': 'cut', 'cut_frame': cut_at}
    return frames, gt

"""Benchmark the DCT backends on the available device.

    python validation/bench_dct.py [--device auto] [--repeats 30]

Reports throughput for `matmul` (cached orthogonal basis, the default) and
`torch_dct` (FFT-based reference), plus the worst-case relative disagreement, so the
`--dct-impl` default can be justified from measurement rather than assumption.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from libs.transforms import dct_2d_matmul, dct_2d_torchdct  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def timeit(fn, blocks, device, repeats: int) -> float:
    for _ in range(5):                       # warm-up (basis cache, kernel autotune)
        fn(blocks)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(blocks)
    sync(device)
    return (time.perf_counter() - t0) / repeats


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='auto')
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--block-size', type=int, default=32)
    p.add_argument('--frames', type=int, default=32, help='1080p frames per batch')
    args = p.parse_args()

    device = pick_device(args.device)
    bs = args.block_size
    n_blocks = (1920 // bs) * (1080 // bs) * args.frames
    blocks = (torch.randn(n_blocks, bs, bs, device=device) * 60.0)
    print(f'device={device.type}  blocks={n_blocks} ({args.frames} frames of 1080p, {bs}x{bs})')

    have_torch_dct = True
    try:
        dct_2d_torchdct(blocks[:1])
    except ImportError as exc:
        have_torch_dct = False
        print(f'torch_dct unavailable: {exc}')

    t_mm = timeit(dct_2d_matmul, blocks, device, args.repeats)
    rows = [('matmul', t_mm)]
    if have_torch_dct:
        rows.append(('torch_dct', timeit(dct_2d_torchdct, blocks, device, args.repeats)))
        a, b = dct_2d_matmul(blocks), dct_2d_torchdct(blocks)
        rel = ((a - b).abs().max() / b.abs().max()).item()
        print(f'max relative disagreement: {rel:.3e} (tolerance 1e-4) '
              f'-> {"PASS" if rel < 1e-4 else "FAIL"}')

    print(f'\n{"impl":<12}{"ms/batch":>12}{"frames/s":>12}{"speedup":>10}')
    for name, t in rows:
        print(f'{name:<12}{t * 1e3:>12.3f}{args.frames / t:>12.1f}'
              f'{t_mm / t:>10.2f}x')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

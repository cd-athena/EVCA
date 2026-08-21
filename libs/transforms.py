import torch
import libs.dct_butterfly_torch as dct_b
import argparse

# Cached DCT-II basis matrices keyed by (size, device, dtype).
# A separable matmul DCT (D @ X @ D^T) beats the FFT-based torch_dct for small
# fixed block sizes (batched GEMM) and, unlike torch.fft, runs natively on MPS.
_DCT_BASIS_CACHE = {}


def _dct_basis(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (n, device, dtype)
    basis = _DCT_BASIS_CACHE.get(key)
    if basis is None:
        k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
        m = torch.arange(n, dtype=torch.float64).unsqueeze(0)
        # Unnormalized DCT-II, identical to torch_dct.dct(x) with norm=None:
        # X[k] = 2 * sum_m x[m] * cos(pi * k * (2m + 1) / (2n))
        basis = 2.0 * torch.cos(torch.pi * k * (2.0 * m + 1.0) / (2.0 * n))
        basis = basis.to(device=device, dtype=dtype)
        _DCT_BASIS_CACHE[key] = basis
    return basis


def dct_2d_matmul(blocks: torch.Tensor) -> torch.Tensor:
    """Separable 2D DCT-II over the last two dimensions via cached basis matmuls."""
    rows = _dct_basis(blocks.shape[-2], blocks.device, blocks.dtype)
    cols = _dct_basis(blocks.shape[-1], blocks.device, blocks.dtype)
    return rows @ blocks @ cols.transpose(0, 1)


def dct_2d_torchdct(blocks: torch.Tensor) -> torch.Tensor:
    """FFT-based reference implementation (the `torch_dct` package).

    Kept as an ablation reference only: it is an optional dependency and, unlike the
    matmul path, torch.fft has no native MPS kernel.
    """
    try:
        import torch_dct
    except ImportError as exc:
        raise ImportError(
            "--dct-impl torch_dct requires the optional 'torch-dct' package "
            "(pip install torch-dct); the default 'matmul' implementation has no "
            "extra dependency."
        ) from exc
    return torch_dct.dct_2d(blocks)


def dct_2d(blocks: torch.Tensor, impl: str = 'matmul') -> torch.Tensor:
    """Separable 2D DCT-II dispatching on the `--dct-impl` selection."""
    if impl == 'matmul':
        return dct_2d_matmul(blocks)
    if impl == 'torch_dct':
        return dct_2d_torchdct(blocks)
    raise ValueError(f'unknown dct-impl: {impl}')


def apply_luma_transform(args: argparse.Namespace, blocks: torch.Tensor, dwt_model=None) -> torch.Tensor:
    """
    Applies the selected discrete transform to the luma blocks.
    """
    if args.transform == 'DWT':
        if dwt_model is None:
            raise ValueError("dwt_model cannot be None when transform is DWT")
        yl, yh = dwt_model(blocks.unsqueeze(1).float())
        yh = yh[0]
        top_row = torch.cat((yl, yh[:, :, 0, :, :]), dim=3)
        bottom_row = torch.cat((yh[:, :, 1, :, :], yh[:, :, 2, :, :]), dim=3)
        return torch.cat((top_row, bottom_row), dim=2)
    elif args.transform == 'DCT_B':
        return dct_b.dct_32_2d(blocks.type(torch.int32))
    else:
        return dct_2d(blocks, getattr(args, 'dct_impl', 'matmul'))


def apply_chroma_transform(args: argparse.Namespace, U_blocks: torch.Tensor, V_blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the selected discrete transform to the chroma blocks.
    Note: DWT is not currently applied to chroma blocks in EVCA.
    """
    if args.transform == 'DCT_B':
        U_DTs = dct_b.dct_16_2d(U_blocks.type(torch.int32))
        V_DTs = dct_b.dct_16_2d(V_blocks.type(torch.int32))
    else:
        impl = getattr(args, 'dct_impl', 'matmul')
        U_DTs = dct_2d(U_blocks, impl)
        V_DTs = dct_2d(V_blocks, impl)
    return U_DTs, V_DTs

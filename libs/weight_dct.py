import torch


def weight_dct(args, device, size=None, keep_dc: bool = False):
    """EVCA's exponential high-frequency DCT weighting.

    `keep_dc=True` retains the DC coefficient's natural formula weight (e^-1) instead
    of zeroing it. Used by `--residual-dc`: for a motion-compensated residual the DC
    term is the block's mean prediction error, which carries real rate cost, unlike
    the DC of an intra block where it is just average brightness.
    """
    if size is None:
        # EVCA uses square blocks, so w = h = args.block_size
        N = args.block_size
    else:
        N = size

    # Create 1-based index tensors for i and j to match the paper's mathematical notation
    i = torch.arange(1, N + 1, dtype=torch.float32, device=device)
    j = torch.arange(1, N + 1, dtype=torch.float32, device=device)

    # Create 2D coordinate grids
    I, J = torch.meshgrid(i, j, indexing='ij')

    # Apply the exponential weighting formula: e^(((i * j) / (w * h))^2 - 1)
    weight_dct = torch.exp(((I * J) / (N * N))**2 - 1.0)

    # Exclude the DC component (where i+j <= 2 in 1-based indexing)
    if not keep_dc:
        weight_dct[0, 0] = 0.0

    return weight_dct

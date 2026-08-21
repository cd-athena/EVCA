"""Motion-compensation strategies.

Every compensator maps a block-resolution motion-vector field onto a pixel-grid
warp of the reference frame. They differ in how the block field becomes a pixel
field (smoothed-bilinear, bilinear, piecewise-constant) and in how overlapping
predictions are blended (OBMC).

MV convention: `mvs` is [B, 2, H_b, W_b] holding (dy, dx) in full-resolution pixels,
with `curr(y, x) ~= ref(y + dy, x + dx)`.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Per-(H, W, device) cache of the normalized base sampling grid and the MV
# normalization constants used by grid_sample. Rebuilding meshgrid/linspace on
# every call costs two kernel launches per frame batch.
_WARP_CACHE: Dict[Tuple[int, int, str], Tuple[torch.Tensor, float, float]] = {}
_GAUSS_CACHE: Dict[str, torch.Tensor] = {}
_OBMC_WEIGHT_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


def warp_constants(H: int, W: int, device: torch.device):
    """Returns (base_grid [1, H, W, 2], x_norm, y_norm) for grid_sample warping."""
    key = (H, W, str(device))
    cached = _WARP_CACHE.get(key)
    if cached is None:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        cached = (base_grid, (W - 1) / 2.0, (H - 1) / 2.0)
        _WARP_CACHE[key] = cached
    return cached


def gauss_kernel(device: torch.device) -> torch.Tensor:
    """3x3 normalized Gaussian, shaped [2, 1, 3, 3] for grouped conv over (dy, dx)."""
    key = str(device)
    weight = _GAUSS_CACHE.get(key)
    if weight is None:
        gauss = torch.tensor([[1.0, 2.0, 1.0],
                              [2.0, 4.0, 2.0],
                              [1.0, 2.0, 1.0]],
                             dtype=torch.float32, device=device) / 16.0
        weight = gauss.repeat(2, 1, 1, 1)
        _GAUSS_CACHE[key] = weight
    return weight


def warp(ref_frame: torch.Tensor, pixel_mvs: torch.Tensor) -> torch.Tensor:
    """Bilinearly resamples `ref_frame` at (y + dy, x + dx) given a pixel-grid MV field.

    Grid coordinates are clamped to [-1, 1], which with align_corners=True is exactly
    equivalent to padding_mode='border' but also works on MPS, where grid_sample's
    border mode is not implemented.
    """
    B, C, H, W = ref_frame.shape
    base_grid, x_norm, y_norm = warp_constants(H, W, ref_frame.device)
    dx = pixel_mvs[:, 1, :, :] / x_norm
    dy = pixel_mvs[:, 0, :, :] / y_norm
    shifted = (base_grid + torch.stack((dx, dy), dim=-1)).clamp_(-1.0, 1.0)
    return F.grid_sample(ref_frame, shifted, mode='bilinear',
                         padding_mode='zeros', align_corners=True)


def smooth_mv_field(mvs: torch.Tensor, mode: str) -> torch.Tensor:
    """Applies `gauss`, `median` (vector median) or `none` smoothing to the MV field."""
    if mode == 'none':
        return mvs.float()
    mvs = mvs.float()
    if mode == 'gauss':
        # Replicate-pad by 1 to prevent boundary shrinkage
        padded = F.pad(mvs, (1, 1, 1, 1), mode='replicate')
        return F.conv2d(padded, gauss_kernel(mvs.device), groups=2)
    if mode == 'median':
        return vector_median(mvs)
    raise ValueError(f'unknown mc-smooth mode: {mode}')


def vector_median(mvs: torch.Tensor) -> torch.Tensor:
    """3x3 vector median filter: picks the neighbourhood vector minimising the sum of
    L2 distances to all others. Unlike a per-component median this always returns a
    vector that actually occurred, so it cannot invent a diagonal MV."""
    B, _, Hb, Wb = mvs.shape
    padded = F.pad(mvs, (1, 1, 1, 1), mode='replicate')
    # [B, 2*9, Hb, Wb] -> [B, 2, 9, Hb*Wb]
    patches = F.unfold(padded, kernel_size=3).view(B, 2, 9, Hb * Wb)
    diff = patches.unsqueeze(3) - patches.unsqueeze(2)      # [B, 2, 9, 9, N]
    dist = torch.sqrt((diff ** 2).sum(dim=1) + 1e-12).sum(dim=2)   # [B, 9, N]
    best = dist.argmin(dim=1, keepdim=True)                  # [B, 1, N]
    picked = torch.gather(patches, 2, best.unsqueeze(1).expand(B, 2, 1, Hb * Wb))
    return picked.view(B, 2, Hb, Wb)


class MotionCompensator(nn.Module):
    """Base class: block-resolution MV field -> motion-compensated frame."""

    def forward(self, ref_frame: torch.Tensor, mvs: torch.Tensor, bs: int) -> torch.Tensor:
        raise NotImplementedError


class DenseMC(MotionCompensator):
    """Continuous MV field: optional smoothing, then bilinear upsample to the pixel grid.

    `align_corners=False` on the upsample registers coarse MV samples to block centres;
    `align_corners=True` would stretch the field and misregister it by up to half a block.
    """

    def __init__(self, smooth: str = 'gauss'):
        super().__init__()
        self.smooth = smooth

    def forward(self, ref_frame: torch.Tensor, mvs: torch.Tensor, bs: int) -> torch.Tensor:
        B, C, H, W = ref_frame.shape
        field = smooth_mv_field(mvs, self.smooth)
        pixel_mvs = F.interpolate(field, size=(H, W), mode='bilinear', align_corners=False)
        return warp(ref_frame, pixel_mvs)


class BlockMC(MotionCompensator):
    """Plain block MC: piecewise-constant MV field (nearest upsample), no blending."""

    def forward(self, ref_frame: torch.Tensor, mvs: torch.Tensor, bs: int) -> torch.Tensor:
        B, C, H, W = ref_frame.shape
        pixel_mvs = F.interpolate(mvs.float(), size=(H, W), mode='nearest')
        return warp(ref_frame, pixel_mvs)


def obmc_weights(bs: int, device: torch.device) -> torch.Tensor:
    """Raised-cosine OBMC blend weights, shaped [5, 1, bs, bs].

    Order is (own, up, down, left, right). Along each axis the two neighbour weights
    are a raised-cosine pair summing to alpha, so own + 2*alpha = 1 at every pixel and
    the blend is a partition of unity (no brightness drift).
    """
    key = (bs, str(device))
    cached = _OBMC_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    alpha = 0.25
    # Pixel centres within the block, normalised to [0, 1]
    t = (torch.arange(bs, device=device, dtype=torch.float32) + 0.5) / bs
    ramp = 0.5 * (1.0 + torch.cos(torch.pi * t))   # 1 at the near edge, 0 at the far edge
    up = alpha * ramp.view(bs, 1).expand(bs, bs)
    down = alpha * (1.0 - ramp).view(bs, 1).expand(bs, bs)
    left = alpha * ramp.view(1, bs).expand(bs, bs)
    right = alpha * (1.0 - ramp).view(1, bs).expand(bs, bs)
    own = torch.full((bs, bs), 1.0 - 2.0 * alpha, device=device)
    w = torch.stack([own, up, down, left, right]).unsqueeze(1)
    _OBMC_WEIGHT_CACHE[key] = w
    return w


class OBMC(MotionCompensator):
    """Overlapped block MC: blends the block's own prediction with those of its four
    neighbours using a raised-cosine window, which suppresses blocking artefacts at
    MV discontinuities."""

    def forward(self, ref_frame: torch.Tensor, mvs: torch.Tensor, bs: int) -> torch.Tensor:
        B, C, H, W = ref_frame.shape
        field = mvs.float()
        padded = F.pad(field, (1, 1, 1, 1), mode='replicate')
        Hb, Wb = field.shape[-2:]
        candidates = [
            field,                              # own
            padded[:, :, 0:Hb, 1:1 + Wb],       # up
            padded[:, :, 2:2 + Hb, 1:1 + Wb],   # down
            padded[:, :, 1:1 + Hb, 0:Wb],       # left
            padded[:, :, 1:1 + Hb, 2:2 + Wb],   # right
        ]
        w = obmc_weights(bs, ref_frame.device)           # [5, 1, bs, bs]
        # Tile each blend weight over the block grid to full resolution
        w_full = w.repeat(1, 1, Hb, Wb)                  # [5, 1, H, W]

        out = torch.zeros_like(ref_frame)
        for i, cand in enumerate(candidates):
            pixel_mvs = F.interpolate(cand, size=(H, W), mode='nearest')
            out = out + warp(ref_frame, pixel_mvs) * w_full[i]
        return out


def build_compensator(mc: str, mc_smooth: str = 'gauss') -> MotionCompensator:
    """Factory for the `--mc` / `--mc-smooth` CLI pair."""
    if mc == 'dense_smooth':
        return DenseMC(smooth=mc_smooth)
    if mc == 'dense':
        return DenseMC(smooth='none')
    if mc == 'block':
        return BlockMC()
    if mc == 'obmc':
        return OBMC()
    raise ValueError(f'unknown mc mode: {mc}')

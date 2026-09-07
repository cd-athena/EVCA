import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from libs.motion_compensation import DenseMC, MotionCompensator


def pooled_pair(current_frame: torch.Tensor, ref_frame: torch.Tensor, pool: int,
                frame_stack: Optional[torch.Tensor] = None):
    """(current, reference) box-filtered down by `pool`.

    `frame_stack` is the tensor the two arguments were sliced out of. EVCA feeds
    overlapping views of one GOP (`frames[1:]` and `frames[:-1]`), so pooling the stack
    once and re-slicing halves the filtering work compared with pooling each side.
    """
    if pool == 1:
        return current_frame, ref_frame
    if frame_stack is not None:
        pooled = F.avg_pool2d(frame_stack, pool, pool)
        return pooled[1:], pooled[:-1]
    return (F.avg_pool2d(current_frame, pool, pool),
            F.avg_pool2d(ref_frame, pool, pool))


@dataclass
class TemporalState:
    """Unified memory footprint (and lazy-evaluation graph) for the metric pipeline.

    With `pool > 1` the frames and `bs` are in the pooled residual domain while `mvs`
    stays in full-resolution pixels, because the exported motion metrics (MVC,
    mean_mv_mag) are defined there and must not change units with the pooling factor.
    `mc_frame` converts on the way into the compensator.
    """
    # Inputs (in the residual domain: pooled by `pool`)
    current_frame: torch.Tensor     #[B, C, H, W]
    ref_frame: torch.Tensor         #[B, C, H, W]
    bs: int = 32                    # block size in the residual domain
    pool: int = 1                   # residual-domain downsampling factor
    # Motion Estimation Outputs
    mvs: torch.Tensor = None        #[B, 2, H_blocks, W_blocks], full-resolution pixels
    sad_map: torch.Tensor = None    #[B, 1, H_blocks, W_blocks]
    # Motion Compensation strategy (defaults to the Gaussian-smoothed dense warp)
    compensator: Optional[MotionCompensator] = None
    # Motion Compensation Outputs (Lazy/Optional)
    _mc_blocks: torch.Tensor = field(default=None, repr=False)  # Unfolded, motion-compensated blocks
    _residual: torch.Tensor = field(default=None, repr=False)   # current_blocks - mc_blocks
    _mc_frame: torch.Tensor = field(default=None, repr=False)   # Full-resolution warped reference

    @property
    def mc_frame(self) -> torch.Tensor:
        """Lazy evaluation of the motion-compensated reference frame."""
        if self._mc_frame is None:
            if self.mvs is None:
                raise ValueError("Motion vectors must be evaluated before extracting mc_blocks")
            compensator = self.compensator if self.compensator is not None else DenseMC('gauss')
            # `mvs` is in full-resolution pixels; the reference is pooled by `pool`.
            mvs = self.mvs if self.pool == 1 else self.mvs / self.pool
            self._mc_frame = compensator(self.ref_frame, mvs, self.bs)
        return self._mc_frame

    @property
    def mc_blocks(self) -> torch.Tensor:
        """Lazy evaluation of Motion-Compensated blocks, unfolded to the block grid."""
        if self._mc_blocks is None:
            self._mc_blocks = self.mc_frame.unfold(2, self.bs, self.bs).unfold(3, self.bs, self.bs).contiguous()
        return self._mc_blocks

    @property
    def residual(self) -> torch.Tensor:
        """Lazy evaluation of the motion-compensated spatial residual.

        Subtracting at frame level and unfolding once is arithmetically identical to
        unfolding both operands and subtracting (unfold is a permutation), but it
        materialises one blocked copy instead of two.
        """
        if self._residual is None:
            self._residual = (self.current_frame - self.mc_frame) \
                .unfold(2, self.bs, self.bs).unfold(3, self.bs, self.bs).contiguous()
        return self._residual

    @property
    def residual_frame(self) -> torch.Tensor:
        """Full-resolution motion-compensated residual [B, C, H, W]."""
        return self.current_frame - self.mc_frame


PATTERN_SHAPES = ('diamond', 'square', 'grid')
# The shapes that are five candidates by construction; `grid` is (2r+1)^2 instead.
SPARSE_PATTERN_SHAPES = ('diamond', 'square')


def quantise_offset(offset: int, pool: int) -> int:
    """`offset` full-resolution pixels expressed as a radius on the 1/`pool` grid.

    Rounds the magnitude and never returns 0, so a pattern always has somewhere to go.
    Note this is deliberately not `offset // pool`: floor division sends -2 to -1 and
    +2 to 0, which would make a pattern reach further one way than the other.
    """
    return max(1, int(round(offset / pool)))


def search_pattern(shape: str, offset: int, pool: int = 1) -> list:
    """Candidate (dy, dx) offsets of a static search pattern, in *pooled* units.

    `offset` is the pattern's reach in full-resolution pixels; at `pool > 1` it is
    quantised onto the pooled grid, so the search's granularity is `pool` pixels and
    its reach stays `offset` pixels. `diamond` puts four neighbours on the axes,
    `square` on the diagonals, and `grid` fills the whole (2r+1)^2 square, which is
    what lets reach and granularity be chosen independently.
    """
    if offset < 1:
        raise ValueError(f'search-pattern offset must be >= 1, got {offset}')
    if pool < 1:
        raise ValueError(f'search pool must be >= 1, got {pool}')
    r = quantise_offset(offset, pool)
    if shape == 'diamond':
        return [(0, 0), (-r, 0), (r, 0), (0, -r), (0, r)]
    if shape == 'square':
        return [(0, 0), (-r, -r), (-r, r), (r, -r), (r, r)]
    if shape == 'grid':
        return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    raise ValueError(f'unknown search pattern: {shape!r}; choose from {PATTERN_SHAPES}')


class PatternBlockMatcher(nn.Module):
    """Static-pattern block matcher, optionally searching on a box-filtered image.

    Every candidate is a whole-frame contiguous slice of one padded reference, so all
    blocks share the shift and a candidate costs one subtract plus one `avg_pool2d`.
    That is the property that keeps the search cheap: cost is linear in the candidate
    count and quadratic in `pool`, and it is lost the moment blocks are given
    individual displacements.

    Searching at `pool > 1` quantises motion vectors to a `pool`-pixel grid but buys
    `pool^2` candidates for the price of one, so reach can be widened at lower cost.
    Decoded vectors are always returned in full-resolution pixels.
    """

    def __init__(self, block_size: int = 32, heuristic: str = 'diamond', offset: int = 2,
                 pool: int = 1):
        super().__init__()
        if pool < 1:
            raise ValueError(f'--me-pool must be >= 1, got {pool}')
        if block_size % pool:
            raise ValueError(f'--me-pool {pool} does not divide --block_size {block_size}')
        if block_size // pool < 4:
            raise ValueError(f'--me-pool {pool} leaves a {block_size // pool}x'
                             f'{block_size // pool} search block; needs at least 4x4')
        self.bs = block_size
        self.pool = pool
        self.search_bs = block_size // pool
        self.pattern = search_pattern(heuristic, offset, pool)
        self.num_cands = len(self.pattern)
        # The pattern's L-infinity reach on the search grid, which is both the padding
        # the reference needs and (times `pool`) the largest MV this search can report.
        self.reach = max(max(abs(dy), abs(dx)) for dy, dx in self.pattern)
        self.reach_px = self.reach * pool

        # Vectorized O(1) lookup table for coordinate decoding, pre-scaled to
        # full-resolution pixels. register_buffer ensures it moves to CUDA/MPS with
        # the module.
        self.register_buffer('pattern_lookup',
                             torch.tensor(self.pattern, dtype=torch.float32) * pool)

    def forward(self, curr_frame: torch.Tensor, ref_frame: torch.Tensor,
                frame_stack: torch.Tensor = None):
        curr, ref = pooled_pair(curr_frame, ref_frame, self.pool, frame_stack)
        B, C, H, W = curr.shape
        bs = self.search_bs
        H_b, W_b = H // bs, W // bs
        R = self.reach

        ref_padded = F.pad(ref, (R, R, R, R), mode='replicate')
        sads = torch.empty((B, self.num_cands, H_b, W_b),
                           device=curr.device, dtype=curr.dtype)

        for idx, (dy, dx) in enumerate(self.pattern):
            # Zero-copy tensor slice, then a hardware-accelerated block SAD.
            ref_slice = ref_padded[:, :, R + dy:H + R + dy, R + dx:W + R + dx]
            abs_diff = torch.abs(curr - ref_slice)
            sads[:, idx] = F.avg_pool2d(abs_diff, bs, bs).squeeze(1)

        # Global hardware reduction: best_idx is [B, H_b, W_b] indexing the pattern.
        best_sad, best_idx = torch.min(sads, dim=1)

        # Advanced indexing expands the index tensor into [B, H_b, W_b, 2]; permute to
        # the [B, 2, H_b, W_b] (dy, dx) layout the metrics and compensators expect.
        mvs = self.pattern_lookup[best_idx].permute(0, 3, 1, 2).contiguous()

        return mvs, best_sad.unsqueeze(1)


# Retired name kept working: the class is no longer sparse-only now that `grid` is a
# pattern shape, but external callers should not have to care.
SparsePatternBlockMatcher = PatternBlockMatcher


class EVCATemporalMetric(nn.Module):
    """Abstract Base Class for Temporal Plugins."""
    def forward(self, state: TemporalState) -> torch.Tensor:
        raise NotImplementedError

class MetricMVC(EVCATemporalMetric):
    """Calculates Motion Vector Field Complexity (Entropy)"""
    def __init__(self):
        super().__init__()
        # fixed Laplacian edge-detector kernel to find spatial variance in the MV field
        laplacian = torch.tensor([[[[0., 1., 0.],
                                    [1., -4., 1.],
                                    [0., 1., 0.]]]])
        self.register_buffer('laplacian_kernel', laplacian.repeat(2, 1, 1, 1))
    
    def forward(self, state: TemporalState) -> torch.Tensor:
        # F.conv2d requires [B, C, H, W] -> mvs are [B, 2, H_b, W_b]
        # Replicate-pad instead of zero-padding: border blocks would otherwise see
        # phantom zero-motion neighbors, producing spurious gradients on global motion.
        mvs_padded = F.pad(state.mvs, (1, 1, 1, 1), mode='replicate')
        mv_gradients = F.conv2d(mvs_padded, self.laplacian_kernel, groups=2)
        # average absolute spatial variance (chaos of the motion field)
        return torch.mean(torch.abs(mv_gradients), dim=[1, 2, 3])

class MetricsTCSAD(EVCATemporalMetric):
    """Outputs the minimum SAD cost of the residual.

    Scored in the search domain, so the value is on a `--me-pool`-dependent scale:
    box-filtering averages detail away before the difference is taken. Values are
    comparable across runs at one pooling factor, not across factors.
    """
    def forward(self, state: TemporalState) -> torch.Tensor:
        return torch.mean(state.sad_map, dim=[1, 2, 3])

class EVCATemporalEngine(nn.Module):
    """Orchestrator for Motion Estimation, Motion Compensation and Metric Plugins."""
    def __init__(self, motion_estimator: nn.Module, metrics: Dict[str, EVCATemporalMetric],
                 compensator: MotionCompensator = None, residual_pool: int = 1):
        super().__init__()
        self.me_module = motion_estimator
        self.metrics = nn.ModuleDict(metrics)
        self.compensator = compensator if compensator is not None else DenseMC('gauss')
        bs = motion_estimator.bs
        if residual_pool < 1 or bs % residual_pool:
            raise ValueError(f'--temporal-pool {residual_pool} does not divide '
                             f'--block_size {bs}')
        if bs // residual_pool < 8:
            raise ValueError(f'--temporal-pool {residual_pool} leaves a '
                             f'{bs // residual_pool}x{bs // residual_pool} residual '
                             f'block; the DCT weighting needs at least 8x8')
        self.residual_pool = residual_pool

    def forward(self, current_frame: torch.Tensor, ref_frame: torch.Tensor,
                frame_stack: torch.Tensor = None) -> Tuple[Dict[str, torch.Tensor], TemporalState]:
        # 1. hardware-accelerated batched ME (own pooling factor)
        mvs, sad_map = self.me_module(current_frame, ref_frame, frame_stack=frame_stack)
        # 2. the residual path runs at its own resolution; MVs stay full-resolution
        rp = self.residual_pool
        curr_r, ref_r = pooled_pair(current_frame, ref_frame, rp, frame_stack)
        state = TemporalState(curr_r, ref_r, bs=self.me_module.bs // rp, pool=rp,
                              compensator=self.compensator)
        state.mvs, state.sad_map = mvs, sad_map
        # 3. evaluate registered plugins dynamically
        results = {}
        for name, metric_module in self.metrics.items():
            results[name] = metric_module(state)
        
        return results, state

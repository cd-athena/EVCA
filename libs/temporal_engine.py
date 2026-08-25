import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from libs.motion_compensation import DenseMC, MotionCompensator


@dataclass
class TemporalState:
    """Unified memory footprint (and lazy-evaluation graph) for the metric pipeline."""
    # Inputs
    current_frame: torch.Tensor     #[B, C, H, W]
    ref_frame: torch.Tensor         #[B, C, H, W]
    bs: int = 32
    # Motion Estimation Outputs
    mvs: torch.Tensor = None        #[B, 2, H_blocks, W_blocks]
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
            self._mc_frame = compensator(self.ref_frame, self.mvs, self.bs)
        return self._mc_frame

    @property
    def mc_blocks(self) -> torch.Tensor:
        """Lazy evaluation of Motion-Compensated blocks, unfolded to the block grid."""
        if self._mc_blocks is None:
            self._mc_blocks = self.mc_frame.unfold(2, self.bs, self.bs).unfold(3, self.bs, self.bs).contiguous()
        return self._mc_blocks
    
    @property
    def residual(self) -> torch.Tensor:
        """Lazy evaluation of the motion-compensated spatial residual."""
        if self._residual is None:
            mc = self.mc_blocks
            curr_blocks = self.current_frame.unfold(2, self.bs, self.bs).unfold(3, self.bs, self.bs).contiguous()
            self._residual = curr_blocks - mc
        return self._residual

    @property
    def residual_frame(self) -> torch.Tensor:
        """Full-resolution motion-compensated residual [B, C, H, W]."""
        return self.current_frame - self.mc_frame
        

PATTERN_SHAPES = ('diamond', 'square')


def search_pattern(shape: str, offset: int) -> list:
    """The five candidate (dy, dx) offsets of a static search pattern, in pixels.

    Every pattern is the collocated block plus four neighbours at `offset` pixels:
    `diamond` puts them on the axes, `square` on the diagonals. The search runs at
    full resolution, so these are literal pixel offsets and every integer motion
    vector the pattern can express is exactly representable.
    """
    if offset < 1:
        raise ValueError(f'search-pattern offset must be >= 1, got {offset}')
    r = offset
    if shape == 'diamond':
        return [(0, 0), (-r, 0), (r, 0), (0, -r), (0, r)]
    if shape == 'square':
        return [(0, 0), (-r, -r), (-r, r), (r, -r), (r, r)]
    raise ValueError(f'unknown search pattern: {shape!r}; choose from {PATTERN_SHAPES}')


class SparsePatternBlockMatcher(nn.Module):
    """Five-point full-resolution block matcher.

    Scores the collocated block and four neighbours at +/- `offset` pixels with SAD and
    keeps the cheapest per block, giving a fixed O(1) search cost. Searching at full
    resolution (rather than on a 2x2-pooled image) costs 4x per candidate but makes odd
    motion vectors representable and matches on unfiltered pixels; with only five
    candidates the total is comparable to a denser half-resolution search.
    """

    def __init__(self, block_size: int = 32, heuristic: str = 'diamond', offset: int = 2):
        super().__init__()
        self.bs = block_size
        self.pattern = search_pattern(heuristic, offset)
        self.num_cands = len(self.pattern)
        # The pattern's L-infinity reach, which is both the padding the reference needs
        # and the largest motion vector this search can report.
        self.reach = offset

        # Vectorized O(1) lookup table for coordinate decoding. register_buffer ensures
        # this tensor moves to CUDA/MPS alongside the module.
        self.register_buffer('pattern_lookup',
                             torch.tensor(self.pattern, dtype=torch.float32))

    def forward(self, curr_frame: torch.Tensor, ref_frame: torch.Tensor):
        B, C, H, W = curr_frame.shape
        H_b, W_b = H // self.bs, W // self.bs
        R = self.reach

        ref_padded = F.pad(ref_frame, (R, R, R, R), mode='replicate')
        sads = torch.empty((B, self.num_cands, H_b, W_b),
                           device=curr_frame.device, dtype=curr_frame.dtype)

        for idx, (dy, dx) in enumerate(self.pattern):
            # Zero-copy tensor slice, then a hardware-accelerated block SAD.
            ref_slice = ref_padded[:, :, R + dy:H + R + dy, R + dx:W + R + dx]
            abs_diff = torch.abs(curr_frame - ref_slice)
            sads[:, idx] = F.avg_pool2d(abs_diff, self.bs, self.bs).squeeze(1)

        # Global hardware reduction: best_idx is [B, H_b, W_b] indexing the pattern.
        best_sad, best_idx = torch.min(sads, dim=1)

        # Advanced indexing expands the index tensor into [B, H_b, W_b, 2]; permute to
        # the [B, 2, H_b, W_b] (dy, dx) layout the metrics and compensators expect.
        mvs = self.pattern_lookup[best_idx].permute(0, 3, 1, 2).contiguous()

        return mvs, best_sad.unsqueeze(1)


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
    """Outputs the minimum SAD cost of the residual."""
    def forward(self, state: TemporalState) -> torch.Tensor:
        return torch.mean(state.sad_map, dim=[1, 2, 3])

class EVCATemporalEngine(nn.Module):
    """Orchestrator for Motion Estimation, Motion Compensation and Metric Plugins."""
    def __init__(self, motion_estimator: nn.Module, metrics: Dict[str, EVCATemporalMetric],
                 compensator: MotionCompensator = None):
        super().__init__()
        self.me_module = motion_estimator
        self.metrics = nn.ModuleDict(metrics)
        self.compensator = compensator if compensator is not None else DenseMC('gauss')

    def forward(self, current_frame: torch.Tensor, ref_frame: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], TemporalState]:
        state = TemporalState(current_frame, ref_frame, bs=self.me_module.bs,
                              compensator=self.compensator)
        # 1. hardware-accelerated batched ME
        state.mvs, state.sad_map = self.me_module(current_frame, ref_frame)
        # 2. evaluate registered plugins dynamically
        results = {}
        for name, metric_module in self.metrics.items():
            results[name] = metric_module(state)
        
        return results, state



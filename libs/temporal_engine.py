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
        

# Search-pattern candidate offsets, in *half-resolution* units: the search runs on a
# 2x2-pooled image, so a candidate (dy, dx) decodes to the full-resolution vector
# (2*dy, 2*dx) and only even-valued MVs are representable.
#
# `diamond_axis` is the Iteration-4 reference pattern and is retained for ablation.
# Despite the name it is a *plus*, not a diamond: every candidate lies on an axis, so
# no diagonal motion is representable at all. Measured against synthetic ground truth,
# a true (4, 4) translation was estimated as (4, 0) -- magnitude 4.00 against a true
# 5.66 -- and `MV_sat_frac`, an L-infinity test, could not see the failure. The
# diagonal-carrying patterns below cut mean MV error on a 12-direction sweep from
# 2.053 px to 1.091 (`diamond`) and 0.911 (`diamond_dense`).
SEARCH_PATTERNS = {
    # 13-point plus, +/- 6 px full-res reach. Iteration-4 reference; no diagonals.
    'diamond_axis': [
        (0, 0),
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-3, 0), (3, 0), (0, -3), (0, 3),
    ],
    # 17-point: the plus plus the four outer diagonals, which make (+/-4, +/-4)
    # full-res exactly representable. Same +/- 6 px L-infinity reach, so MV_sat_frac
    # stays comparable with `diamond_axis`.
    'diamond': [
        (0, 0),
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-3, 0), (3, 0), (0, -3), (0, 3),
        (-2, -2), (-2, 2), (2, -2), (2, 2),
    ],
    # 21-point: adds the inner diagonals, making (+/-2, +/-2) full-res representable
    # as well. Lowest MV error of the three; costs four more candidates.
    'diamond_dense': [
        (0, 0),
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-2, -2), (-2, 2), (2, -2), (2, 2),
        (-3, 0), (3, 0), (0, -3), (0, 3),
    ],
    # 9-point sparse square, radius 2 at half-res -> +/- 4 px full-res.
    # NOTE: eight of its nine candidates sit at L-infinity == the pattern's max reach,
    # so `MV_sat_frac` reads ~1.0 by construction and carries no information here.
    'square': [
        (0, 0),
        (-2, 0), (2, 0), (0, -2), (0, 2),    # Cardinal directions
        (-2, -2), (-2, 2), (2, -2), (2, 2),  # The Diagonals
    ],
}


class SparsePatternBlockMatcher(nn.Module):
    """
    Sparse Pattern Block Matcher (The 'Fixed Diamond').
    Evaluates a static, deterministic pattern of motion vectors to achieve
    fast O(1) search complexity while preserving highly accurate heuristics.
    """
    def __init__(self, block_size: int = 32, heuristic: str = 'diamond', dilation: int = 1):
        super().__init__()
        self.bs = block_size
        self.bs_c = block_size // 2     # coarse block size

        if heuristic not in SEARCH_PATTERNS:
            raise ValueError(f"unknown heuristic pattern: {heuristic}")
        base_pattern = SEARCH_PATTERNS[heuristic]

        # 1.c apply resolution-aware dilation: multiply offsets by dilation factor to 
        # stretch the search horizon for large resolutions
        self.pattern = [(dy * dilation, dx * dilation) for dy, dx in base_pattern]
        self.num_cands = len(self.pattern)
        
        # Calculate padding dynamically based on the pattern's maximum reach
        self.R_c = max(max(abs(dy), abs(dx)) for dy, dx in self.pattern)

        # Maximum decodable full-resolution MV magnitude (L-inf). A block whose MV
        # reaches this bound sits on the search-pattern boundary (used for MV_sat_frac).
        self.max_reach_fullres = float(2 * self.R_c)

        # 2. Vectorized O(1) Lookup Table for Coordinate Decoding
        # The search runs at half resolution (2x2 avg_pool), so we pre-multiply by 2.0
        # to decode candidate offsets into full-resolution (even-valued) vectors.
        # register_buffer ensures this tensor automatically moves to MPS/CUDA alongside the model.
        lookup_tensor = torch.tensor(self.pattern, dtype=torch.float32) * 2.0
        self.register_buffer('pattern_lookup', lookup_tensor)

    def forward(self, curr_frame: torch.Tensor, ref_frame: torch.Tensor):
        B, C, H, W = curr_frame.shape
        H_b, W_b = H // self.bs, W // self.bs

        # =====================================================================
        # COARSE SEARCH: Shift-and-Pool over the Fixed Diamond
        # =====================================================================
        curr_c = F.avg_pool2d(curr_frame, kernel_size=2, stride=2)
        ref_c = F.avg_pool2d(ref_frame, kernel_size=2, stride=2)
        
        ref_c_padded = F.pad(ref_c, (self.R_c, self.R_c, self.R_c, self.R_c), mode='replicate')

        # Pre-allocate SAD tensor for exactly 13 candidates instead of 49
        sads = torch.empty((B, self.num_cands, H_b, W_b), device=curr_frame.device, dtype=curr_frame.dtype)

        for idx, (dy, dx) in enumerate(self.pattern):
            # Zero-copy tensor slice
            ref_slice = ref_c_padded[:, :, self.R_c+dy : (H//2)+self.R_c+dy, self.R_c+dx : (W//2)+self.R_c+dx]
            
            # Hardware-accelerated block SAD
            abs_diff = torch.abs(curr_c - ref_slice)
            sads[:, idx] = F.avg_pool2d(abs_diff, kernel_size=self.bs_c, stride=self.bs_c).squeeze(1)

        # Global Hardware Reduction
        # best_idx is a 3D tensor of shape [B, H_b, W_b] containing values 0-12
        best_sad, best_idx = torch.min(sads, dim=1) 

        # =====================================================================
        # COORDINATE DECODING: Vectorized Advanced Indexing
        # =====================================================================
        # We pass the entire batch's index tensor into the 2D lookup table. 
        # PyTorch advanced indexing automatically expands this into shape [B, H_b, W_b, 2]
        decoded_mvs = self.pattern_lookup[best_idx]
        
        # Split the vectors and reshape to [B, 1, H_b, W_b] to match EVCA plugin formats
        best_dy = decoded_mvs[..., 0].unsqueeze(1)
        best_dx = decoded_mvs[..., 1].unsqueeze(1)
        
        best_mv = torch.cat([best_dy, best_dx], dim=1)

        return best_mv, best_sad.unsqueeze(1)

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



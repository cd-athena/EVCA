import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Tuple

# Per-(H, W, device) cache of the normalized base sampling grid and MV
# normalization constants used by grid_sample warping. Rebuilding
# meshgrid/linspace on every call costs two kernel launches per frame batch.
_WARP_CACHE: Dict[Tuple[int, int, str], Tuple[torch.Tensor, float, float]] = {}

# Per-device cache of the fixed 3x3 Gaussian smoothing kernel (2 groups).
_GAUSS_CACHE: Dict[str, torch.Tensor] = {}


def _warp_constants(H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, float, float]:
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


def _gauss_kernel(device: torch.device) -> torch.Tensor:
    key = str(device)
    weight = _GAUSS_CACHE.get(key)
    if weight is None:
        gauss = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0]
        ], dtype=torch.float32, device=device) / 16.0
        weight = gauss.repeat(2, 1, 1, 1)
        _GAUSS_CACHE[key] = weight
    return weight


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
    # Motion Compensation Outputs (Lazy/Optional)
    _mc_blocks: torch.Tensor = field(default=None, repr=False)  # Unfolded, motion-compensated blocks
    _residual: torch.Tensor = field(default=None, repr=False)   # current_blocks - mc_blocks
    
    @property
    def mc_blocks(self) -> torch.Tensor:
        """Lazy evaluation of Motion-Compensated blocks with field smoothing."""
        if self._mc_blocks is None:
            if self.mvs is None:
                raise ValueError("Motion vectors must be evaluated before extracting mc_blocks")

            B, C, H, W = self.ref_frame.shape

            # 1. Coarse Vector Field Gaussian Smoothing (3x3 Kernel)
            # Replicate-pad by 1 to prevent boundary shrinkage
            mvs_padded = F.pad(self.mvs.float(), (1, 1, 1, 1), mode='replicate')
            weight = _gauss_kernel(self.ref_frame.device)
            mvs_smooth = F.conv2d(mvs_padded, weight, groups=2)

            # 2. Continuous Bilinear Upsampling to Pixel Grid
            # align_corners=False: coarse MV samples represent block centers, not corner
            # pixels; True would stretch the field and misregister it by up to half a block.
            pixel_mvs = F.interpolate(mvs_smooth, size=(H, W), mode='bilinear', align_corners=False)

            # 3. Normalized Sampling Grid Construction [-1, 1] (cached per (H, W, device))
            base_grid, x_norm, y_norm = _warp_constants(H, W, self.ref_frame.device)

            # Normalize pixel MVs to grid space [-1, 1]
            dx = pixel_mvs[:, 1, :, :] / x_norm
            dy = pixel_mvs[:, 0, :, :] / y_norm
            normalized_mvs = torch.stack((dx, dy), dim=-1)

            # Clamp to [-1, 1]: with align_corners=True this is exactly equivalent to
            # padding_mode='border' (same pixel-coordinate clamp), but works on MPS,
            # where grid_sample's border mode is not implemented.
            shifted_grid = (base_grid + normalized_mvs).clamp_(-1.0, 1.0)

            # 4. Differentiable Bilinear Image Warping
            mc_frame = F.grid_sample(
                self.ref_frame,
                shifted_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )

            # 5. Vectorized Unfold into 32x32 Blocks
            self._mc_blocks = mc_frame.unfold(2, self.bs, self.bs).unfold(3, self.bs, self.bs).contiguous()

        return self._mc_blocks
    
    @property
    def residual(self) -> torch.Tensor:
        """Lazy evaluation of the motion-compensated spatial residual."""
        if self._residual is None:
            mc = self.mc_blocks
            curr_blocks = self.current_frame.unfold(2, self.bs, self.bs).unfold(3, self.bs, self.bs).contiguous()
            self._residual = curr_blocks - mc
        return self._residual
        

class SparsePatternBlockMatcher(nn.Module):
    """
    Sparse Pattern Block Matcher (The 'Fixed Diamond').
    Evaluates a static, deterministic diamond of motion vectors to achieve
    fast O(1) search complexity while preserving highly accurate heuristics.
    """
    def __init__(self, block_size: int = 32, heuristic: str = 'diamond', dilation: int = 1):
        super().__init__()
        self.bs = block_size
        self.bs_c = block_size // 2     # coarse block size

        if heuristic == 'diamond':
        # 1.a 13-point Large Diamond Pattern (Half-Resolution Offsets)
        # The search runs on a 2x2-pooled image, so offsets are half-res and the
        # effective full-res search radius is +/- 6 pixels (even-valued MVs only).
            base_pattern = [
                (0, 0),
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-2, 0), (2, 0), (0, -2), (0, 2),
                (-3, 0), (3, 0), (0, -3), (0, 3)
            ]
        elif heuristic == 'square':
        # 1.b 9-point Sparse Square (Radius = 2 at half-res -> Effective +/- 4 pixels)
        # Captures center, cross, and extreme diagonals.
            base_pattern = [
                (0, 0),
                (-2, 0), (2, 0), (0, -2), (0, 2),    # Cardinal directions
                (-2, -2), (-2, 2), (2, -2), (2, 2)   # The Diagonals (1,1), (-1,-1), etc.
            ]
        else:
            raise ValueError(f"unknown heuristic pattern: {heuristic}")
        
        # 1.c apply resolution-aware dilation: multiply offsets by dilation factor to 
        # stretch the search horizon for large resolutions
        self.pattern = [(dy * dilation, dx * dilation) for dy, dx in base_pattern]
        self.num_cands = len(self.pattern)
        
        # Calculate padding dynamically based on the pattern's maximum reach
        self.R_c = max(max(abs(dy), abs(dx)) for dy, dx in self.pattern)

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
    """Orchestrator for Motion Estimation and Metric Plugins."""
    def __init__(self, motion_estimator: nn.Module, metrics: Dict[str, EVCATemporalMetric]):
        super().__init__()
        self.me_module = motion_estimator
        self.metrics = nn.ModuleDict(metrics)
    
    def forward(self, current_frame: torch.Tensor, ref_frame: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], TemporalState]:
        state = TemporalState(current_frame, ref_frame, bs=self.me_module.bs)
        # 1. hardware-accelerated batched ME
        state.mvs, state.sad_map = self.me_module(current_frame, ref_frame)
        # 2. evaluate registered plugins dynamically
        results = {}
        for name, metric_module in self.metrics.items():
            results[name] = metric_module(state)
        
        return results, state



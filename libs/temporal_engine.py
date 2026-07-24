import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict

@dataclass
class TemporalState:
    """Unified memory footprint for the metric pipeline."""
    # Inputs
    current_frame: torch.Tensor     #[B, C, H, W]
    ref_frame: torch.Tensor         #[B, C, H, W]
    # Motion Estimation Outputs
    mvs: torch.Tensor = None        #[2, H_blocks, W_blocks]
    sad_map: torch.Tensor = None    #[1, H_blocks, W_blocks]
    # Motion Compensation Outputs (Lazy/Optional)
    mc_blocks: torch.Tensor = None  # Unfolded, motion-compensated blocks
    residual: torch.Tensor = None   # current_blocks - mc_blocks

class SparsePatternBlockMatcher(nn.Module):
    """
    Sparse Pattern Block Matcher (The 'Fixed Diamond').
    Evaluates a static, deterministic diamond of motion vectors to achieve
    fast O(1) search complexity while preserving highly accurate heuristics.
    """
    def __init__(self, block_size: int = 32, heuristic: str = 'diamond'):
        super().__init__()
        self.bs = block_size
        self.bs_c = block_size // 2     # coarse block size

        if heuristic == 'diamond':
        # 1.a 13-point Large Diamond Pattern (Quarter-Resolution Offsets)
        # This gives us a highly efficient +/- 6 pixel effective search radius.
            self.pattern = [
                (0, 0),
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-2, 0), (2, 0), (0, -2), (0, 2),
                (-3, 0), (3, 0), (0, -3), (0, 3)
            ]
        elif heuristic == 'square':
        # 1.b 9-point Sparse Square (Radius = 2 at quarter-res -> Effective +/- 4 pixels)
        # Captures center, cross, and extreme diagonals.
            self.pattern = [
                (0, 0),
                (-2, 0), (2, 0), (0, -2), (0, 2),    # Cardinal directions
                (-2, -2), (-2, 2), (2, -2), (2, 2)   # The Diagonals (1,1), (-1,-1), etc.
            ]
        else:
            raise ValueError(f"unknown heuristic pattern: {heuristic}")
        
        self.num_cands = len(self.pattern)
        
        # Calculate padding dynamically based on the pattern's maximum reach
        self.R_c = max(max(abs(dy), abs(dx)) for dy, dx in self.pattern)

        # 2. Vectorized O(1) Lookup Table for Coordinate Decoding
        # We pre-multiply by 2.0 so the lookup table outputs full-resolution vectors instantly.
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
        mv_gradients = F.conv2d(state.mvs, self.laplacian_kernel, groups=2, padding=1)
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
    
    def forward(self, current_frame: torch.Tensor, ref_frame: torch.Tensor):
        state = TemporalState(current_frame, ref_frame)
        # 1. hardware-accelerated batched ME
        state.mvs, state.sad_map = self.me_module(current_frame, ref_frame)
        # 2. evaluate registered plugins dynamically
        results = {}
        for name, metric_module in self.metrics.items():
            results[name] = metric_module(state)
        
        return results



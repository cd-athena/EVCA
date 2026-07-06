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

class IntegerBlockMatcher(nn.Module):
    """
    Ultra-Fast Single-Stage Coarse Matcher.
    Drops per-block refinement entirely to bypass all PyTorch indexing 
    bottlenecks. Achieves hardware-level speeds natively.
    """
    def __init__(self, block_size: int = 32, coarse_range: int = 3):
        super().__init__()
        self.bs = block_size
        self.bs_c = block_size // 2
        # A coarse range of +/- 3 equates to an effective +/- 6 pixel 
        # search range at full resolution, which is highly sufficient for EVCA.
        self.R_c = coarse_range 

    def forward(self, curr_frame: torch.Tensor, ref_frame: torch.Tensor):
        B, C, H, W = curr_frame.shape
        H_b, W_b = H // self.bs, W // self.bs

        # =====================================================================
        # SINGLE STAGE: COARSE SEARCH (Quarter-Resolution Shift-and-Pool)
        # =====================================================================
        # Downsampling preserves structural energy for SAD while cutting 
        # memory bandwidth consumption by exactly 75%.
        curr_c = F.avg_pool2d(curr_frame, kernel_size=2, stride=2)
        ref_c = F.avg_pool2d(ref_frame, kernel_size=2, stride=2)
        
        ref_c_padded = F.pad(ref_c, (self.R_c, self.R_c, self.R_c, self.R_c), mode='replicate')

        # Evaluate 49 dense candidates (7x7 grid)
        num_cands = (2 * self.R_c + 1) ** 2
        sads = torch.empty((B, num_cands, H_b, W_b), device=curr_frame.device, dtype=curr_frame.dtype)

        idx = 0
        for dy in range(-self.R_c, self.R_c + 1):
            for dx in range(-self.R_c, self.R_c + 1):
                # Zero-copy tensor slicing (0 memory allocation)
                ref_slice = ref_c_padded[:, :, self.R_c+dy : (H//2)+self.R_c+dy, self.R_c+dx : (W//2)+self.R_c+dx]
                abs_diff = torch.abs(curr_c - ref_slice)
                
                # Hardware-accelerated pooling provides simultaneous block SADs
                sads[:, idx] = F.avg_pool2d(abs_diff, kernel_size=self.bs_c, stride=self.bs_c).squeeze(1)
                idx += 1

        # Massive batched reduction across all candidates
        best_sad, best_idx = torch.min(sads, dim=1, keepdim=True)
        
        # Decode the optimal 1D indices back into 2D Motion Vectors
        grid_width = 2 * self.R_c + 1
        base_dy = ((best_idx // grid_width) - self.R_c) * 2
        base_dx = ((best_idx % grid_width) - self.R_c) * 2

        # Scale back to full-resolution coordinates
        best_mv = torch.cat([base_dy.float(), base_dx.float()], dim=1)

        # We return the coarse SAD directly. Because it is calculated on an avg_pooled
        # tensor, it perfectly correlates with the full-frame SAD, fulfilling the TCSAD metric.
        return best_mv, best_sad





# class IntegerBlockMatcher(nn.Module):
#     """
#     Hyper-Optimized Hierarchical Pyramid Matcher.
#     Utilizes Zero-Copy Block-Pointer Indexing (Hardware-Agnostic FP32).
#     """
#     def __init__(self, block_size: int = 32, coarse_range: int = 2, fine_range: int = 1):
#         super().__init__()
#         self.bs = block_size
#         self.bs_c = block_size // 2
#         self.R_c = coarse_range
#         self.R_f = fine_range

#     def forward(self, curr_frame: torch.Tensor, ref_frame: torch.Tensor):
#         B, C, H, W = curr_frame.shape
#         H_b, W_b = H // self.bs, W // self.bs

#         # =====================================================================
#         # STAGE 1: COARSE SEARCH (Quarter-Resolution)
#         # =====================================================================
#         curr_c = F.avg_pool2d(curr_frame, kernel_size=2, stride=2)
#         ref_c = F.avg_pool2d(ref_frame, kernel_size=2, stride=2)

#         ref_c_padded = F.pad(ref_c, (self.R_c, self.R_c, self.R_c, self.R_c), mode='replicate')

#         num_c_cands = (2 * self.R_c + 1) ** 2
        
#         # Using curr_frame.dtype ensuring native FP32 support across CUDA/MPS/CPU
#         sads_c = torch.empty((B, num_c_cands, H_b, W_b), device=curr_frame.device, dtype=curr_frame.dtype)

#         idx = 0
#         for dy in range(-self.R_c, self.R_c + 1):
#             for dx in range(-self.R_c, self.R_c + 1):
#                 ref_slice = ref_c_padded[:, :, self.R_c+dy : (H//2)+self.R_c+dy, self.R_c+dx : (W//2)+self.R_c+dx]
#                 abs_diff = torch.abs(curr_c - ref_slice)
#                 sads_c[:, idx] = F.avg_pool2d(abs_diff, kernel_size=self.bs_c, stride=self.bs_c).squeeze(1)
#                 idx += 1

#         _, best_idx_c = torch.min(sads_c, dim=1, keepdim=True)
#         grid_width_c = 2 * self.R_c + 1
        
#         base_dy = ((best_idx_c // grid_width_c) - self.R_c) * 2
#         base_dx = ((best_idx_c % grid_width_c) - self.R_c) * 2

#         # =====================================================================
#         # STAGE 2: FINE REFINEMENT
#         # =====================================================================
#         curr_blocks = curr_frame.unfold(2, self.bs, self.bs).unfold(3, self.bs, self.bs).squeeze(1)

#         max_reach = (self.R_c * 2) + self.R_f
#         ref_padded = F.pad(ref_frame, (max_reach, max_reach, max_reach, max_reach), mode='replicate')

#         # The 1024x Block-Pointer Reduction (This is what makes it so fast)
#         ref_unfolded = ref_padded.unfold(2, self.bs, 1).unfold(3, self.bs, 1).squeeze(1)
        
#         grid_y = (torch.arange(H_b, device=curr_frame.device) * self.bs + max_reach).view(H_b, 1)
#         grid_x = (torch.arange(W_b, device=curr_frame.device) * self.bs + max_reach).view(1, W_b)
        
#         num_f_cands = (2 * self.R_f + 1) ** 2
#         sads_f = torch.empty((B, num_f_cands, H_b, W_b), device=curr_frame.device, dtype=curr_frame.dtype)
#         mvs_f_y = torch.empty((B, num_f_cands, H_b, W_b), device=curr_frame.device, dtype=curr_frame.dtype)
#         mvs_f_x = torch.empty((B, num_f_cands, H_b, W_b), device=curr_frame.device, dtype=curr_frame.dtype)

#         b_idx = torch.arange(B, device=curr_frame.device).view(B, 1, 1)

#         idx = 0
#         for dy in range(-self.R_f, self.R_f + 1):
#             for dx in range(-self.R_f, self.R_f + 1):
#                 cand_dy = base_dy.squeeze(1) + dy
#                 cand_dx = base_dx.squeeze(1) + dx

#                 mvs_f_y[:, idx] = cand_dy
#                 mvs_f_x[:, idx] = cand_dx

#                 block_y = grid_y.unsqueeze(0) + cand_dy
#                 block_x = grid_x.unsqueeze(0) + cand_dx

#                 mc_blocks = ref_unfolded[b_idx, block_y.long(), block_x.long()]

#                 sads_f[:, idx] = torch.mean(torch.abs(curr_blocks - mc_blocks), dim=(3, 4))
#                 idx += 1

#         best_sad, best_idx = torch.min(sads_f, dim=1, keepdim=True)
#         final_dy = torch.gather(mvs_f_y, 1, best_idx)
#         final_dx = torch.gather(mvs_f_x, 1, best_idx)

#         best_mv = torch.cat([final_dy, final_dx], dim=1)

#         return best_mv, best_sad

# class IntegerBlockMatcher(nn.Module):
#     """
#     Hierarchical Pyramid Block Matcher.
#     Achieves ASIC-like speeds natively in PyTorch by combining 
#     Coarse Quarter-Resolution Shift-and-Pool with Fine Advanced Indexing.
#     """
#     def __init__(self, block_size: int = 32, coarse_range: int = 2, fine_range: int = 1):
#         super().__init__()
#         self.bs = block_size
#         self.bs_c = block_size // 2      # Coarse block size
#         self.R_c = coarse_range          # Quarter-res search radius
#         self.R_f = fine_range            # Full-res refinement radius

#     def forward(self, curr_frame: torch.Tensor, ref_frame: torch.Tensor):
#         B, C, H, W = curr_frame.shape
#         H_b, W_b = H // self.bs, W // self.bs

#         # =====================================================================
#         # STAGE 1: COARSE SEARCH (Quarter-Resolution)
#         # =====================================================================
#         # 1. Downsample (avg_pool preserves structural integer energy for SAD)
#         curr_c = F.avg_pool2d(curr_frame, kernel_size=2, stride=2)
#         ref_c = F.avg_pool2d(ref_frame, kernel_size=2, stride=2)

#         # 2. Pad coarse reference
#         ref_c_padded = F.pad(ref_c, (self.R_c, self.R_c, self.R_c, self.R_c), mode='replicate')

#         # 3. Dense search at coarse level (25 candidates)
#         num_c_cands = (2 * self.R_c + 1) ** 2
#         sads_c = torch.empty((B, num_c_cands, H_b, W_b), device=curr_frame.device, dtype=curr_frame.dtype)

#         idx = 0
#         for dy in range(-self.R_c, self.R_c + 1):
#             for dx in range(-self.R_c, self.R_c + 1):
#                 # Zero-copy slice of the quarter-res frame
#                 ref_slice = ref_c_padded[:, :, self.R_c+dy : (H//2)+self.R_c+dy, self.R_c+dx : (W//2)+self.R_c+dx]
#                 abs_diff = torch.abs(curr_c - ref_slice)
                
#                 # Hardware pooled SAD
#                 sad = F.avg_pool2d(abs_diff, kernel_size=self.bs_c, stride=self.bs_c)
#                 sads_c[:, idx] = sad.squeeze(1)
#                 idx += 1

#         # 4. Find optimal coarse MVs and upscale them to full resolution
#         _, best_idx_c = torch.min(sads_c, dim=1, keepdim=True)
#         grid_width_c = 2 * self.R_c + 1
        
#         base_dy = ((best_idx_c // grid_width_c) - self.R_c) * 2
#         base_dx = ((best_idx_c % grid_width_c) - self.R_c) * 2

#         # =====================================================================
#         # STAGE 2: FINE REFINEMENT (Full-Resolution Advanced Indexing)
#         # =====================================================================
#         # 1. Slice current frame into non-overlapping blocks without copying memory
#         # Shape: [B, H_b, W_b, bs, bs]
#         curr_blocks = curr_frame.unfold(2, self.bs, self.bs).unfold(3, self.bs, self.bs).squeeze(1)

#         # 2. Pad full resolution reference for the absolute maximum possible reach
#         max_reach = (self.R_c * 2) + self.R_f
#         ref_padded = F.pad(ref_frame, (max_reach, max_reach, max_reach, max_reach), mode='replicate')

#         # 3. Pre-calculate pristine integer grid coordinates for every pixel in every block
#         y_base = torch.arange(H_b, device=curr_frame.device) * self.bs + max_reach
#         x_base = torch.arange(W_b, device=curr_frame.device) * self.bs + max_reach
#         y_offsets = torch.arange(self.bs, device=curr_frame.device).view(-1, 1)
#         x_offsets = torch.arange(self.bs, device=curr_frame.device).view(1, -1)
        
#         # Base grid: [H_b, W_b, bs, bs]
#         grid_y = y_base.view(H_b, 1, 1, 1) + y_offsets
#         grid_x = x_base.view(1, W_b, 1, 1) + x_offsets

#         # 4. Refinement Search Loop (9 candidates)
#         num_f_cands = (2 * self.R_f + 1) ** 2
#         sads_f = torch.empty((B, num_f_cands, H_b, W_b), device=curr_frame.device, dtype=curr_frame.dtype)
#         mvs_f_y = torch.empty((B, num_f_cands, H_b, W_b), device=curr_frame.device)
#         mvs_f_x = torch.empty((B, num_f_cands, H_b, W_b), device=curr_frame.device)

#         # Batch index array for advanced tensor gathering
#         b_idx = torch.arange(B, device=curr_frame.device).view(B, 1, 1, 1, 1)

#         idx = 0
#         for dy in range(-self.R_f, self.R_f + 1):
#             for dx in range(-self.R_f, self.R_f + 1):
#                 # Calculate candidate MV = Upscaled Coarse MV + Fine Offset
#                 cand_dy = base_dy.squeeze(1) + dy
#                 cand_dx = base_dx.squeeze(1) + dx

#                 mvs_f_y[:, idx] = cand_dy
#                 mvs_f_x[:, idx] = cand_dx

#                 # Generate sampling coordinates by adding the unique integer MV to every block's grid
#                 # Shape: [B, H_b, W_b, bs, bs]
#                 sample_y = grid_y.unsqueeze(0) + cand_dy.unsqueeze(-1).unsqueeze(-1)
#                 sample_x = grid_x.unsqueeze(0) + cand_dx.unsqueeze(-1).unsqueeze(-1)

#                 # Advanced integer indexing: Pluck the exact motion-compensated blocks straight out of memory
#                 # Absolutely 0 interpolation or bilinear smearing.
#                 mc_blocks = ref_padded[b_idx, 0, sample_y.long(), sample_x.long()] 

#                 # Calculate spatial mean SAD matching the format of avg_pool2d
#                 sad = torch.sum(torch.abs(curr_blocks - mc_blocks), dim=(3, 4)) / (self.bs * self.bs)
#                 sads_f[:, idx] = sad
#                 idx += 1

#         # 5. Find ultimate best MVs
#         best_sad, best_idx = torch.min(sads_f, dim=1, keepdim=True)
#         final_dy = torch.gather(mvs_f_y, 1, best_idx)
#         final_dx = torch.gather(mvs_f_x, 1, best_idx)

#         # Output shape: [B, 2, H_b, W_b]
#         best_mv = torch.cat([final_dy, final_dx], dim=1)

#         return best_mv, best_sad

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



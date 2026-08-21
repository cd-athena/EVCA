"""Motion-estimation strategies.

Both estimators share the interface consumed by `EVCATemporalEngine`:

    mvs, sad_map = estimator(curr_frame, ref_frame)

where `mvs` is [B, 2, H_b, W_b] holding (dy, dx) in full-resolution pixels and
`sad_map` is [B, 1, H_b, W_b] holding the winning block cost, with the convention
`curr(y, x) ~= ref(y + dy, x + dx)`.

`HierarchicalBlockMatcher` walks a resolution pyramid: an exhaustive search at the
coarsest level fixes the large-displacement part, and each finer level upsamples the
field and corrects it by a small radius. All candidates of a level are evaluated by
stacking shifted references along the channel dimension and issuing a single
`avg_pool2d`, so the number of kernel launches is independent of the candidate count.
"""
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Peak bytes allowed for the [b, K, H, W] candidate-difference tensor of one level.
# Larger frame batches are split into chunks that each stay under this budget; the
# candidate axis is never split, so every chunk is still one batched pooling op.
_SAD_BUDGET_BYTES = 512 * 1024 * 1024


def _offsets_square(radius: int) -> List[Tuple[int, int]]:
    """Every integer offset in the (2r+1)^2 square, centre first."""
    offs = [(0, 0)]
    offs += [(dy, dx)
             for dy in range(-radius, radius + 1)
             for dx in range(-radius, radius + 1)
             if (dy, dx) != (0, 0)]
    return offs


def batched_block_sad(curr: torch.Tensor, ref: torch.Tensor,
                      offsets: Sequence[Tuple[int, int]], block: int) -> torch.Tensor:
    """Block-mean absolute difference for every candidate offset.

    `curr`/`ref` are [B, 1, H, W]; the result is [B, K, H_b, W_b]. The shifted
    references are stacked on the channel axis so a single `avg_pool2d` reduces all
    candidates at once. The frame batch is split into chunks that keep the
    [b, K, H, W] intermediates inside `_SAD_BUDGET_BYTES`; the candidate axis is never
    split, so each chunk remains one batched pooling op.
    """
    B, _, H, W = curr.shape
    K = len(offsets)
    reach = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
    ref_padded = F.pad(ref, (reach, reach, reach, reach), mode='replicate')

    # Two K-sized intermediates are alive at once (the stack and the difference).
    chunk = max(1, int(_SAD_BUDGET_BYTES // max(1, 2 * K * H * W * 4)))
    out = []
    for start in range(0, B, chunk):
        stop = min(B, start + chunk)
        shifted = torch.cat(
            [ref_padded[start:stop, :, reach + dy:reach + dy + H,
                        reach + dx:reach + dx + W] for dy, dx in offsets], dim=1)
        diff = (curr[start:stop] - shifted).abs_()
        out.append(F.avg_pool2d(diff, kernel_size=block, stride=block))
    return torch.cat(out, dim=0)


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


class HierarchicalBlockMatcher(nn.Module):
    """Pyramid search: exhaustive at the coarsest level, +/- refine_radius per level.

    Levels are powers-of-two downscales of the luma plane. The coarse level is searched
    exhaustively within `coarse_radius` (in that level's pixels), which sets the total
    reach at `coarse_radius * coarsest_scale` full-resolution pixels. Each finer level
    doubles the field and corrects it within `refine_radius`, so the reachable
    correction around the coarse estimate is `sum(refine_radius * scale)` and the final
    vectors are integer-pel rather than the even-only vectors of the sparse pattern.
    """

    def __init__(self, block_size: int = 32, width: int = 1920,
                 coarse_radius: int = 8, refine_radius: int = 1,
                 max_range: int = None):
        super().__init__()
        self.bs = block_size
        self.refine_radius = refine_radius

        # Pyramid at 1/4, 1/2 and full resolution; 4K and wider get a 1/8 level so the
        # coarse search still reaches far enough without an enormous candidate count.
        self.scales = [8, 4, 2, 1] if width >= 3840 else [4, 2, 1]
        self.coarse_scale = self.scales[0]

        if max_range is not None:
            coarse_radius = max(1, int(math.ceil(max_range / self.coarse_scale)))
        self.coarse_radius = coarse_radius

        self.coarse_offsets = _offsets_square(coarse_radius)
        self.refine_offsets = _offsets_square(refine_radius)

        # Full-resolution reach of the exhaustive coarse stage, plus what the
        # refinement levels can add on top.
        refine_reach = sum(refine_radius * s for s in self.scales[1:])
        self.max_reach_fullres = float(coarse_radius * self.coarse_scale + refine_reach)

        self.register_buffer('coarse_lookup',
                             torch.tensor(self.coarse_offsets, dtype=torch.float32))
        self.register_buffer('refine_lookup',
                             torch.tensor(self.refine_offsets, dtype=torch.float32))

    def _level_inputs(self, frame: torch.Tensor, scale: int) -> torch.Tensor:
        if scale == 1:
            return frame
        return F.avg_pool2d(frame, kernel_size=scale, stride=scale)

    def forward(self, curr_frame: torch.Tensor, ref_frame: torch.Tensor):
        B, _, H, W = curr_frame.shape

        mvs = None          # [B, 2, Hb, Wb] in *current level* pixel units
        sad = None
        for level, scale in enumerate(self.scales):
            curr_l = self._level_inputs(curr_frame, scale)
            ref_l = self._level_inputs(ref_frame, scale)
            block_l = max(1, self.bs // scale)

            if mvs is None:
                # Coarsest level: exhaustive search about the origin.
                sads = batched_block_sad(curr_l, ref_l, self.coarse_offsets, block_l)
                sad, best = torch.min(sads, dim=1)
                decoded = self.coarse_lookup[best]                    # [B, Hb, Wb, 2]
                mvs = decoded.permute(0, 3, 1, 2).contiguous()
            else:
                # Finer level: the field doubles with the resolution, then is corrected.
                mvs = mvs * 2.0
                sads = batched_block_sad(curr_l, ref_l, self.refine_offsets, block_l,
                                         centers=mvs)
                sad, best = torch.min(sads, dim=1)
                delta = self.refine_lookup[best].permute(0, 3, 1, 2).contiguous()
                mvs = mvs + delta

        return mvs, sad.unsqueeze(1)


def build_motion_estimator(args, width: int) -> nn.Module:
    """Constructs the motion estimator selected by the `--me*` flags.

    Options accepted by the parser but not yet implemented raise here rather than
    silently degrading to the default search, so an ablation can never report a
    variant it did not actually run.
    """
    unimplemented = []
    if args.me_subpel != 0:
        unimplemented.append(f'--me-subpel {args.me_subpel}')
    if args.me_predictor != 'none':
        unimplemented.append(f'--me-predictor {args.me_predictor}')
    if args.me_lambda != 0.0:
        unimplemented.append(f'--me-lambda {args.me_lambda}')
    if args.me_merge:
        unimplemented.append('--me-merge')
    if args.me_criterion != 'sad':
        unimplemented.append(f'--me-criterion {args.me_criterion}')
    if unimplemented:
        raise NotImplementedError(
            'not implemented yet (Phase 3): ' + ', '.join(unimplemented))

    if args.me == 'hierarchical':
        return HierarchicalBlockMatcher(
            block_size=args.block_size,
            width=width,
            coarse_radius=getattr(args, 'me_coarse_radius', 8),
            refine_radius=getattr(args, 'me_refine_radius', 1),
        )

    dilation_factor = max(1, width // 1920)  # 1080p has multiplier of 1
    return SparsePatternBlockMatcher(
        block_size=args.block_size,
        heuristic=args.heuristic,
        dilation=dilation_factor,
    )

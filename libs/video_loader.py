import numpy as np
import torch
import argparse
from typing import Tuple, List, Optional
from libs.colorfulness import calculate_hasler_suesstrunk_colorfulness_yuv

def load_gop(args: argparse.Namespace, stream, start_frame: int, end_frame: int, device: torch.device, 
             width: int, height: int, pix_size: float, luma_size: int, chroma_size: int, 
             uv_w: int, uv_h: int, cb_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], List[float]]:
    """
    Reads the Y, U, and V frames sequentially for a GOP (Group of Pictures) and extracts block-level tensors 
    along with frame-level colorfulness metrics if enabled.
    
    Returns:
        Y_blocks: Tensor of shape [Total_Blocks, block_size, block_size]
        U_blocks: Tensor of shape [Total_Blocks, cb_size, cb_size] or None
        V_blocks: Tensor of shape [Total_Blocks, cb_size, cb_size] or None
        colorfulness_batch: List of colorfulness floats
    """
    Y_blocks_list = []
    U_blocks_list = []
    V_blocks_list = []
    colorfulness_batch = []
    
    frames = np.arange(start_frame, end_frame, args.sample_rate)
    
    for frame in frames:
        # Seek to frame start
        y_offset = int(frame * width * height * pix_size)
        stream.seek(y_offset)
        
        # Read Y
        Y = np.fromfile(stream, dtype=np.uint8, count=luma_size).reshape(height, width)
        Y_t = torch.from_numpy(
            Y[:height // args.block_size * args.block_size, :width // args.block_size * args.block_size]
        ).to(device, non_blocking=True)
        
        b = Y_t.unfold(0, args.block_size, args.block_size).unfold(1, args.block_size, args.block_size).contiguous().view(-1, args.block_size, args.block_size)
        Y_blocks_list.append(b)
        
        # Read U and V if needed
        if args.chroma_complexity or args.colorfulness:
            U = np.fromfile(stream, dtype=np.uint8, count=chroma_size).reshape(uv_h, uv_w)
            V = np.fromfile(stream, dtype=np.uint8, count=chroma_size).reshape(uv_h, uv_w)
            
            if args.colorfulness:
                # Edge case safety
                if len(U.flatten()) < chroma_size or len(V.flatten()) < chroma_size:
                    colorfulness_batch.append(0.0)
                else:
                    colorfulness_val = calculate_hasler_suesstrunk_colorfulness_yuv(U, V, bit_depth=8)
                    colorfulness_batch.append(colorfulness_val)
                    
            if args.chroma_complexity:
                if args.pix_fmt != 'yuv420':
                    raise NotImplementedError("Chroma energy currently optimized only for YUV4:2:0")
                
                U_t = torch.from_numpy(
                    U[:uv_h // cb_size * cb_size, :uv_w // cb_size * cb_size]
                ).to(device, non_blocking=True).float()
                V_t = torch.from_numpy(
                    V[:uv_h // cb_size * cb_size, :uv_w // cb_size * cb_size]
                ).to(device, non_blocking=True).float()
                
                u_b = U_t.unfold(0, cb_size, cb_size).unfold(1, cb_size, cb_size).contiguous().view(-1, cb_size, cb_size)
                v_b = V_t.unfold(0, cb_size, cb_size).unfold(1, cb_size, cb_size).contiguous().view(-1, cb_size, cb_size)
                
                U_blocks_list.append(u_b)
                V_blocks_list.append(v_b)
                
    Y_blocks = torch.cat(Y_blocks_list, dim=0)
    
    U_blocks = None
    V_blocks = None
    if args.chroma_complexity and len(U_blocks_list) > 0:
        U_blocks = torch.cat(U_blocks_list, dim=0)
        V_blocks = torch.cat(V_blocks_list, dim=0)
        
    return Y_blocks, U_blocks, V_blocks, colorfulness_batch

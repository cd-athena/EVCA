import argparse
import numpy as np
import torch
from typing import Tuple

def extract_chroma_blocks(args: argparse.Namespace, stream, start: int, end: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read and unfold the U and V planes into discrete blocks.
    Implemented only for 4:2:0 subsampling (4:4:4 will follow).
    
    Returns:
        U_blocks, V_blocks: Tensors of shape [Total_Blocks, cb_size, cb_size]
    """
    width, height = map(int, args.resolution.split('x'))
    
    # 4:2:0 subsampling: chroma planes are half width and half height
    uv_w, uv_h = width // 2, height // 2
    cb_size = args.block_size // 2  # chroma block size is half of luma
    
    frames = np.arange(start, end, args.sample_rate)
    u_blocks_list, v_blocks_list = [], []
    
    for frame in frames:
        if args.pix_fmt == 'yuv420':
            # frame size is W*H*1.5 - jump directly past luma to the U plane
            y_offset = int(frame * width * height * 1.5)
            stream.seek(y_offset + (width * height))
        else:
            raise NotImplementedError("Chroma energy currently optimized only for YUV4:2:0")
        
        # read U and V bytes
        uv_count = uv_w * uv_h
        U = np.fromfile(stream, dtype=np.uint8, count=uv_count).reshape(uv_h, uv_w)
        V = np.fromfile(stream, dtype=np.uint8, count=uv_count).reshape(uv_h, uv_w)
        
        # use GPU via torch
        U_t = torch.from_numpy(
            U[:uv_h // cb_size * cb_size, :uv_w // cb_size * cb_size]
        ).to(device, non_blocking=True).float()
        V_t = torch.from_numpy(
            V[:uv_h // cb_size * cb_size, :uv_w // cb_size * cb_size]
        ).to(device, non_blocking=True).float()
        
        # sliding window via stride manipulation (unfold) to get U, V blocks
        u_b = U_t.unfold(0, cb_size, cb_size).unfold(1, cb_size, cb_size).contiguous().view(-1, cb_size, cb_size)
        v_b = V_t.unfold(0, cb_size, cb_size).unfold(1, cb_size, cb_size).contiguous().view(-1, cb_size, cb_size)
        
        u_blocks_list.append(u_b)
        v_blocks_list.append(v_b)

    return torch.cat(u_blocks_list, dim=0), torch.cat(v_blocks_list, dim=0)
        
        
    
    
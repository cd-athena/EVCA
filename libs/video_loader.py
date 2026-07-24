import numpy as np
import torch
import mmap
import os
import argparse
from typing import Tuple, List, Optional
from libs.colorfulness import calculate_hasler_suesstrunk_colorfulness_yuv

def load_gop(args: argparse.Namespace, stream, start_frame: int, end_frame: int, device: torch.device, 
             width: int, height: int, pix_size: float, luma_size: int, chroma_size: int, 
             uv_w: int, uv_h: int, cb_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], List[float]]:
    """
    Reads the Y, U, and V frames sequentially for a GOP (Group of Pictures) and extracts block-level tensors 
    along with frame-level colorfulness metrics if enabled.
    
    Safe for massive files, but bottlenecks on disk I/O.
    
    Returns:
        Y_blocks: Tensor of shape [Total_Blocks, block_size, block_size]
        U_blocks: Tensor of shape [Total_Blocks, cb_size, cb_size] or None
        V_blocks: Tensor of shape [Total_Blocks, cb_size, cb_size] or None
        colorfulness_batch: List of colorfulness floats
        Y_frames: Tensore of shape [Batch, 1, Height, Width] (Float) for Mition Estimation
    """
    Y_blocks_list = []
    U_blocks_list = []
    V_blocks_list = []
    colorfulness_batch = []
    Y_frames_list = []
    
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
        
        Y_frames_list.append(Y_t.unsqueeze(0).unsqueeze(0).float()) # store full frame before slicing, adding [Batch, Channel] dims and casting to float
        
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
    
    # concatenate all stored frames into a single batch tensor of shape [B, 1, H, W]
    Y_frames = torch.cat(Y_frames_list, dim=0)
    
    U_blocks = None
    V_blocks = None
    if args.chroma_complexity and len(U_blocks_list) > 0:
        U_blocks = torch.cat(U_blocks_list, dim=0)
        V_blocks = torch.cat(V_blocks_list, dim=0)
        
    return Y_blocks, U_blocks, V_blocks, colorfulness_batch, Y_frames

def load_gop_optimized(args: argparse.Namespace, stream, start_frame: int, end_frame: int, device: torch.device, 
             width: int, height: int, pix_size: float, luma_size: int, chroma_size: int, 
             uv_w: int, uv_h: int, cb_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], List[float]]:
    """
    Optimized Memory-Mapped I/O loader. Maximizes throughput via zero-copy views.
    Automatically degrades gracefully if CUDA is not present.
    
    Important: will lead to Out-Of-Memory errors on very large video files.
    """
    # Calculate the exact byte range needed for this GOP
    frames = np.arange(start_frame, end_frame, args.sample_rate)
    num_frames = len(frames)
    frame_byte_size = int(width * height * pix_size)
    
    # 1. Memory-Map the file (Zero-copy I/O)
    # this allows to treat the massive file on disk as a numpy array in RAM
    fd = stream.fileno()
    file_size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_COPY)
    
    try:
        # HARDWARE CHECK: only pin memory if a NVIDIA GPU is being used.
        # Apple MPS (Unified Memory) or CPU-only modes will bypass this safely.
        is_cuda = (device.type == 'cuda')
        
        # 2. pre-allocate Pinned Memory (page-locked CPU RAM for fast PCIe DMA transfer)
        Y_batch_cpu = torch.empty((num_frames, height, width), dtype=torch.uint8, pin_memory=is_cuda)
        
        U_batch_cpu, V_batch_cpu = None, None
        colorfulness_batch = []
        if args.chroma_complexity or args.colorfulness:
            U_batch_cpu = torch.empty((num_frames, uv_h, uv_w), dtype=torch.uint8, pin_memory=is_cuda)
            V_batch_cpu = torch.empty((num_frames, uv_h, uv_w), dtype=torch.uint8, pin_memory=is_cuda)
        
        # 3. Populate pinned memory directly from the Memory Map
        for i, frame in enumerate(frames):
            y_offset = frame * frame_byte_size
            
            # Extract Y (Luma)
            Y_view = np.frombuffer(mm, dtype=np.uint8, count=luma_size, offset=y_offset).reshape(height, width)
            Y_batch_cpu[i].copy_(torch.from_numpy(Y_view))
            
            # Extract U and V (Chroma)
            if U_batch_cpu is not None:
                u_offset = y_offset + luma_size
                v_offset = u_offset + chroma_size
                
                U_view = np.frombuffer(mm, dtype=np.uint8, count=chroma_size, offset=u_offset).reshape(uv_h, uv_w)
                V_view = np.frombuffer(mm, dtype=np.uint8, count=chroma_size, offset=v_offset).reshape(uv_h, uv_w)
                
                U_batch_cpu[i].copy_(torch.from_numpy(U_view))
                V_batch_cpu[i].copy_(torch.from_numpy(V_view))
                
                if args.colorfulness:
                    colorfulness_val = calculate_hasler_suesstrunk_colorfulness_yuv(U_view, V_view, bit_depth=8)
                    colorfulness_batch.append(colorfulness_val)
        
        # Delete the zero-copy views to release the C-level exported buffer pointers. 
        # This guarantees the mmap can close safely.
        if 'Y_view' in locals(): del Y_view
        if 'U_view' in locals(): del U_view
        if 'V_view' in locals(): del V_view
            
        # 4. Asynchronous DMA transfer to GPU
        Y_gpu = Y_batch_cpu.to(device, non_blocking=True, dtype=torch.float32)
        
        # 5. Global tensor manipulation
        # Crop to block-size boundaries to guarantee strict interface parity with standard loader
        crop_h = (height // args.block_size) * args.block_size
        crop_w = (width // args.block_size) * args.block_size
        Y_gpu_cropped = Y_gpu[:, :crop_h, :crop_w]
        
        # Shape: [B, H_cropped, W_cropped] -> [B, 1, H_cropped, W_cropped]
        Y_frames = Y_gpu_cropped.unsqueeze(1)
        
        # 6. Vectorized Unfolding Across the Batch
        Y_blocks = Y_gpu_cropped.unfold(1, args.block_size, args.block_size).unfold(2, args.block_size, args.block_size)
        Y_blocks = Y_blocks.contiguous().view(-1, args.block_size, args.block_size)
        
        U_blocks, V_blocks = None, None
        if args.chroma_complexity:
            U_gpu = U_batch_cpu.to(device, non_blocking=True, dtype=torch.float32)
            V_gpu = V_batch_cpu.to(device, non_blocking=True, dtype=torch.float32)
            
            crop_uv_h = (uv_h // cb_size) * cb_size
            crop_uv_w = (uv_w // cb_size) * cb_size
            U_gpu_cropped = U_gpu[:, :crop_uv_h, :crop_uv_w]
            V_gpu_cropped = V_gpu[:, :crop_uv_h, :crop_uv_w]
            
            U_blocks = U_gpu_cropped.unfold(1, cb_size, cb_size).unfold(2, cb_size, cb_size).contiguous().view(-1, cb_size, cb_size)
            V_blocks = V_gpu_cropped.unfold(1, cb_size, cb_size).unfold(2, cb_size, cb_size).contiguous().view(-1, cb_size, cb_size)
    
    finally:
        # clean up memory map
        mm.close()
    
    return Y_blocks, U_blocks, V_blocks, colorfulness_batch, Y_frames
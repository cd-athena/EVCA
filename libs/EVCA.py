import argparse
import os
from pathlib import Path

import numpy as np
import torch
from pytorch_wavelets import DWTForward

from libs.feature_extraction import feature_extraction, temporal_feature_extraction, chroma_energy_extraction
from libs.plot_block_info_EVCA import plot_block_info_EVCA
from libs.write_block_info import write_block_info
from libs.plot_frame_metrics_EVCA import plot_frame_metrics_EVCA
from libs.weight_dct import weight_dct
from libs.video_loader import load_gop, load_gop_optimized
from libs.transforms import apply_luma_transform, apply_chroma_transform
from libs.exporter import export_features_to_csv
from libs.temporal_engine import EVCATemporalEngine, MetricMVC, MetricsTCSAD, SparsePatternBlockMatcher

def EVCA(args: argparse.Namespace, input_list, device) -> None:
    width = int(args.resolution.split('x')[0])
    height = int(args.resolution.split('x')[1])

    bytes_per_sample = 1 if args.bit_depth == 8 else 2
    if args.pix_fmt == 'yuv420':
        pix_size = 1.5 * bytes_per_sample
    elif args.pix_fmt == 'yuv444':
        pix_size = 3.0 * bytes_per_sample
    else:
        raise ValueError(f"Unsupported pixel format '{args.pix_fmt}'. Supported formats are 'yuv420' and 'yuv444'.")

    steps = args.gopsize
    
    # Pre-compute the DCT weight tensor (32x32)
    cached_weights_dct = weight_dct(args, device)

    for file in input_list:
        number_of_frames = int(Path(file).stat().st_size // (width * height * pix_size))
        nframes = args.frames if args.frames != 0 else number_of_frames
        steps = steps if steps <= nframes else nframes
        args.input = file
        stream = open(args.input, 'rb')

        last_energy = torch.tensor([], device=device)
        last_SC = torch.tensor([], device=device)

        need_block_info = bool(args.block_info or args.plot_info)

        out_frames = [[] for _ in range(5)] if args.colorfulness else [[] for _ in range(4)]
        out_blocks = [[] for _ in range(4)]
        
        out_frames_u = []
        out_frames_v = []
        out_blocks_u = []
        out_blocks_v = []
        
        out_blocks_sad = []
        out_blocks_mv = []
        out_blocks_tcmc = []
        
        temporal_engine = None
        out_mvc = []
        out_tcsad = []
        out_tcmc = []
        
        if args.motion_estimation:
            dilation_factor = max(1, width // 1920) # 1080p has multiplier of 1
            me_module = SparsePatternBlockMatcher(
                block_size=args.block_size,
                heuristic=args.heuristic,
                dilation=dilation_factor
            ).to(device)
            metrics = {
                'mvc': MetricMVC().to(device),
                'tc_sad': MetricsTCSAD().to(device)
            }
            temporal_engine = EVCATemporalEngine(me_module, metrics).to(device)
            # if hasattr(torch, "compile"):
            #     temporal_engine = torch.compile(temporal_engine, mode="reduce-overhead")
        last_Y_frame = None
        dwt = None
        if args.transform == 'DWT':
            dwt = DWTForward().to(device)
            
        luma_size = width * height
        if args.pix_fmt == 'yuv420':
            chroma_size = (width // 2) * (height // 2)
            uv_w, uv_h = width // 2, height // 2
            cb_size = args.block_size // 2
        elif args.pix_fmt == 'yuv444':
            chroma_size = width * height
            uv_w, uv_h = width, height
            cb_size = args.block_size
            
        if args.chroma_complexity:
            chroma_weights = weight_dct(args, device, size=cb_size)

        for f in range(0, nframes, steps):
            actual_num_frames = len(range(f, min(nframes, f + steps), args.sample_rate))
            
            # Load Data
            loader_function = load_gop_optimized if args.loader == 'optimized' else load_gop
            Y_blocks, U_blocks, V_blocks, colorfulness_batch, Y_frames = loader_function(
                args, stream, f, min(nframes, f + steps), device, 
                width, height, pix_size, luma_size, chroma_size, uv_w, uv_h, cb_size
            )
            
            # Luma Processing
            DTs = apply_luma_transform(args, Y_blocks, dwt_model=dwt)

            B_blocks, SC_blocks, energy = feature_extraction(args, DTs, actual_num_frames, device, cached_weights_dct)
            TC_blocks, TC2_blocks = temporal_feature_extraction(args, f, SC_blocks, energy, last_SC, last_energy)

            last_energy = energy[-2:]
            last_SC = SC_blocks[-2:]
            
            # Motion Estimation
            if temporal_engine is not None and Y_frames.shape[0] > 0:
                if last_Y_frame is not None:
                    me_input_frames = torch.cat([last_Y_frame, Y_frames], dim=0)
                else:
                    me_input_frames = Y_frames
                
                if me_input_frames.shape[0] > 1:
                    current_frames = me_input_frames[1:]
                    ref_frames = me_input_frames[:-1]
                    # batched ME and metrics
                    me_results, me_state = temporal_engine(current_frames, ref_frames)
                    mvc_batch = me_results['mvc'].cpu().numpy().ravel()
                    tcsad_batch = me_results['tc_sad'].cpu().numpy().ravel()
                    
                    if need_block_info:
                        sad_map_flat = me_state.sad_map.squeeze(1).reshape(current_frames.shape[0], -1)
                        out_blocks_sad.append(sad_map_flat.detach())
                        mv_mag = torch.sqrt(me_state.mvs[:, 0]**2 + me_state.mvs[:, 1]**2)
                        mv_flat = mv_mag.reshape(current_frames.shape[0], -1)
                        out_blocks_mv.append(mv_flat.detach())
                    
                    tcmc_batch = None
                    tc_uncomp_batch = None
                    if args.profile == 'full':
                        # evaluate motion-compensated Residual (TC_MC)
                        # extract spatial residual dynamically
                        # Spatial residual tensor of shape: [B, 1, H_blocks, W_blocks, 32, 32]
                        residual_tensor = me_state.residual
                        # reshape 6D block tensor into 3D block stack: [Total_32x32_Blocks, 32, 32]
                        residual_flat = residual_tensor.view(-1, args.block_size, args.block_size)
                        DTs_mc = apply_luma_transform(args, residual_flat, dwt_model=dwt)
                        # extract high-frequency weighted energy (mimicks EVCA)
                        _, SC_blocks_mc, _ = feature_extraction(args, DTs_mc, current_frames.shape[0], device, cached_weights_dct)
                        
                        # intra-mode energy gating
                        curr_SC_blocks = SC_blocks[1:] if f == 0 else SC_blocks
                        SC_blocks_mc = torch.minimum(SC_blocks_mc, curr_SC_blocks)
                        
                        if need_block_info:
                            out_blocks_tcmc.append(SC_blocks_mc.detach())
                        # collapse block energies into frame-level TC_MC score
                        num_blocks = (width // args.block_size) * (height // args.block_size)
                        tcmc_frame = SC_blocks_mc.sum(dim=1) / num_blocks
                        tcmc_batch = tcmc_frame.cpu().numpy().ravel()
                        
                    if f == 0:
                        # pad first frame with 0 (since it has no reference)
                        out_mvc.extend([0.0] + list(mvc_batch))
                        out_tcsad.extend([0.0] + list(tcsad_batch))
                        if tcmc_batch is not None:
                            out_tcmc.extend([0.0] + list(tcmc_batch))
                    else:
                        out_mvc.extend(list(mvc_batch))
                        out_tcsad.extend(list(tcsad_batch))
                        if tcmc_batch is not None:
                            out_tcmc.extend(list(tcmc_batch))
                last_Y_frame = Y_frames[-1:]
                
            # Chroma Processing
            if args.chroma_complexity and U_blocks is not None:
                U_DTs, V_DTs = apply_chroma_transform(args, U_blocks, V_blocks)
                # unpack tuple
                SC_u_blocks, SC_v_blocks = chroma_energy_extraction(args, U_DTs, V_DTs, actual_num_frames, chroma_weights)
                # calculate frame-level complexity
                block_count = (width // args.block_size) * (height // args.block_size)
                SC_u_frame = SC_u_blocks.sum(dim=[1]) / block_count
                SC_v_frame = SC_v_blocks.sum(dim=[1]) / block_count
                
                SC_u_frame = SC_u_frame.cpu().numpy().ravel()
                SC_v_frame = SC_v_frame.cpu().numpy().ravel()
                
                out_frames_u.extend(SC_u_frame)
                out_frames_v.extend(SC_v_frame)
                out_blocks_u.extend(SC_u_blocks)
                out_blocks_v.extend(SC_v_blocks)
            


            # Aggregation
            B_frame = B_blocks.mean(dim=1)
            SC_frame = SC_blocks.sum(dim=[1]) / ((width // args.block_size) * (height // args.block_size))
            TC_frame = TC_blocks.sum(dim=[1]) / ((width // args.block_size) * (height // args.block_size))
            TC2_frame = TC2_blocks.sum(dim=[1]) / ((width // args.block_size) * (height // args.block_size))

            B_frame = B_frame.cpu().numpy().ravel()
            SC_frame = SC_frame.cpu().numpy().ravel()
            TC_frame = TC_frame.cpu().numpy().ravel()
            TC2_frame = TC2_frame.cpu().numpy().ravel()
            
            if f == 0:
                TC_frame = np.insert(TC_frame, 0, 0)
                TC2_frame = np.insert(TC2_frame, 0, 0)
                if len(np.arange(0, steps, args.sample_rate)) > 1:
                    TC2_frame = np.insert(TC2_frame, 0, 0)

            out_frames[0].extend(B_frame)
            out_frames[1].extend(SC_frame)
            out_frames[2].extend(TC_frame)
            out_frames[3].extend(TC2_frame)

            out_blocks[0].extend(B_blocks)
            out_blocks[1].extend(SC_blocks)
            out_blocks[2].extend(TC_blocks)
            out_blocks[3].extend(TC2_blocks)
            
            if args.colorfulness:
                out_frames[4].extend(colorfulness_batch)

        stream.close()
    
        # Bit-Depth Normalization (Amplitude Scaling)
        # Brings 10-bit and 12-bit metrics down to an 8-bit equivalent scale.
        if args.bit_depth > 8:
            bit_scale = 2 ** (args.bit_depth - 8)
            
            # Scale Luma Complexity
            out_frames[1] = (np.array(out_frames[1]) / bit_scale).tolist()  # SC
            out_frames[2] = (np.array(out_frames[2]) / bit_scale).tolist()  # TC
            out_frames[3] = (np.array(out_frames[3]) / bit_scale).tolist()  # TC2
            
            # Scale Chroma Complexity
            if args.chroma_complexity:
                out_frames_u = (np.array(out_frames_u) / bit_scale).tolist()
                out_frames_v = (np.array(out_frames_v) / bit_scale).tolist()
                
            # Scale Spatial Residuals
            if args.motion_estimation:
                out_tcsad = (np.array(out_tcsad) / bit_scale).tolist()
                if args.profile == 'full':
                    out_tcmc = (np.array(out_tcmc) / bit_scale).tolist()

        # Export CSV
        final_csv_path = export_features_to_csv(args, file, out_frames,
            out_frames_u=out_frames_u if args.chroma_complexity else None, 
            out_frames_v=out_frames_v if args.chroma_complexity else None,
            out_mvc=out_mvc if args.motion_estimation else None,
            out_tcsad=out_tcsad if args.motion_estimation else None,
            out_tcmc=out_tcmc if (args.motion_estimation and args.profile == 'full') else None
        )
        # Additional block plotting / metrics
        if args.block_info == 0 and args.plot_info == 1:
            args.block_info = 1
            print("To plot features we set -bp 1.")

        number_of_frames = int(Path(file).stat().st_size // (width * height * pix_size))
        if args.block_info:
            write_block_info(
                args, out_blocks[0], out_blocks[1], out_blocks[2], out_blocks[3], number_of_frames,
                SC_u=out_blocks_u if args.chroma_complexity else None,
                SC_v=out_blocks_v if args.chroma_complexity else None,
                sad_blocks=out_blocks_sad if args.motion_estimation else None,
                mv_blocks=out_blocks_mv if args.motion_estimation else None,
                tcmc_blocks=out_blocks_tcmc if (args.motion_estimation and args.profile == 'full') else None
            )

        if args.plot_info:
            plot_block_info_EVCA(args, number_of_frames)
        if args.plot_metrics:
            plot_frame_metrics_EVCA(args, final_csv_path)

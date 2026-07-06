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
from libs.weight_dct import weight_dct_by_size
from libs.video_loader import load_gop
from libs.transforms import apply_luma_transform, apply_chroma_transform
from libs.exporter import export_features_to_csv
from libs.temporal_engine import EVCATemporalEngine, IntegerBlockMatcher, MetricMVC, MetricsTCSAD

def EVCA(args: argparse.Namespace, input_list, device) -> None:
    width = int(args.resolution.split('x')[0])
    height = int(args.resolution.split('x')[1])

    pix_size = 1.5
    if args.pix_fmt == 'yuv420':
        pix_size = 1.5
    elif args.pix_fmt == 'yuv444':
        pix_size = 3

    steps = args.gopsize
    for file in input_list:
        number_of_frames = int(Path(file).stat().st_size // (width * height * pix_size))
        nframes = args.frames if args.frames != 0 else number_of_frames
        steps = steps if steps <= nframes else nframes
        args.input = file
        stream = open(args.input, 'rb')

        last_energy = torch.tensor([], device=device)
        last_SC = torch.tensor([], device=device)

        out_frames = [[] for _ in range(5)] if args.colorfulness else [[] for _ in range(4)]
        out_blocks = [[] for _ in range(4)]
        
        out_frames_u = []
        out_frames_v = []
        out_blocks_u = []
        out_blocks_v = []
        
        temporal_engine = None
        out_mvc = []
        out_tcsad = []
        if args.motion_estimation:
            me_module = IntegerBlockMatcher(args.block_size, args.search_range).to(device)
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
            chroma_weights = weight_dct_by_size(cb_size, device)

        for f in range(0, nframes, steps):
            actual_num_frames = len(range(f, min(nframes, f + steps), args.sample_rate))
            
            # Load Data
            Y_blocks, U_blocks, V_blocks, colorfulness_batch, Y_frames = load_gop(
                args, stream, f, min(nframes, f + steps), device, 
                width, height, pix_size, luma_size, chroma_size, uv_w, uv_h, cb_size
            )
            
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
                    me_results = temporal_engine(current_frames, ref_frames)
                    mvc_batch = me_results['mvc'].cpu().numpy().ravel()
                    tcsad_batch = me_results['tc_sad'].cpu().numpy().ravel()
                    
                    if f == 0:
                        # pad first frame with 0 (since it has no reference)
                        out_mvc.extend([0.0] + list(mvc_batch))
                        out_tcsad.extend([0.0] + list(tcsad_batch))
                    else:
                        out_mvc.extend(list(mvc_batch))
                        out_tcsad.extend(list(tcsad_batch))
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
            
            # Luma Processing
            DTs = apply_luma_transform(args, Y_blocks, dwt_model=dwt)

            B_blocks, SC_blocks, energy = feature_extraction(args, DTs, actual_num_frames, device)
            TC_blocks, TC2_blocks = temporal_feature_extraction(args, f, SC_blocks, energy, last_SC, last_energy)

            last_energy = energy[-2:]
            last_SC = SC_blocks[-2:]

            # Aggregation
            B_frame = B_blocks.mean(dim=1)
            SC_frame = SC_blocks.sum(dim=[1]) / ((width // args.block_size) * (height // args.block_size))
            TC_frame = TC_blocks.sum(dim=[1]) / ((width // args.block_size) * (height // args.block_size))
            TC2_frame = TC2_blocks.sum(dim=[1]) / ((width // args.block_size) * (height) // args.block_size)

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

        # Export CSV
        final_csv_path = export_features_to_csv(args, file, out_frames,
            out_frames_u=out_frames_u if args.chroma_complexity else None, 
            out_frames_v=out_frames_v if args.chroma_complexity else None,
            out_mvc=out_mvc if args.motion_estimation else None,
            out_tcsad=out_tcsad if args.motion_estimation else None)
        
        # Additional block plotting / metrics
        if args.block_info == 0 and args.plot_info == 1:
            args.block_info = 1
            print("To plot features we set -bp 1.")

        number_of_frames = int(Path(file).stat().st_size // (width * height * pix_size))
        if args.block_info:
            if args.chroma_complexity:
                write_block_info(args, out_blocks[0], out_blocks[1], out_blocks[2], out_blocks[3], number_of_frames, out_blocks_u, out_frames_v)
            else:
                write_block_info(args, out_blocks[0], out_blocks[1], out_blocks[2], out_blocks[3], number_of_frames)
        if args.plot_info:
            plot_block_info_EVCA(args, number_of_frames)
        if args.plot_metrics:
            plot_frame_metrics_EVCA(args, final_csv_path)

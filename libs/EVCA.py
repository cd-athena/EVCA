import argparse
import os
import torch
import numpy as np
import pandas as pd
import torch_dct as dct
import libs.dct_butterfly_torch as dct_b
from pytorch_wavelets import DWTForward
from libs.frame_to_block import frame_to_block
from libs.feature_extraction import feature_extraction, temporal_feature_extraction
from libs.write_block_info import write_block_info
from libs.plot_block_info_EVCA import plot_block_info_EVCA
from pathlib import Path


def EVCA(args: argparse.Namespace, input_list, device) -> None:
    width            = int(args.resolution.split('x')[0])
    height           = int(args.resolution.split('x')[1])

    pix_size = 1.5
    if args.pix_fmt == 'yuv420':
        pix_size = 1.5
    elif args.pix_fmt == 'yuv444':
        pix_size = 3

    steps = args.gopsize
    for file in input_list:
        number_of_frames = Path(file).stat().st_size // (width * height * pix_size)
        nframes = args.frames if args.frames != 0 else number_of_frames
        steps = steps if steps <= nframes else nframes
        args.input = file
        stream = open(args.input, 'rb')

        last_energy = torch.tensor([], device=device)
        last_SC = torch.tensor([], device=device)

        out_frames = [[] for _ in range(4)]
        out_blocks = [[] for _ in range(4)]

        for f in range(0, nframes, steps):
            blocks = frame_to_block(args, stream, f, min(nframes, f+steps), device)
            if args.transform == 'DWT':
                dwt = DWTForward().to(device)
                yl, yh = dwt(blocks.unsqueeze(1).float())
                yh = yh[0]
                top_row = torch.cat((yl, yh[:, :, 0, :, :]), dim=3)
                bottom_row = torch.cat((yh[:, :, 1, :, :], yh[:, :, 2, :, :]), dim=3)
                DTs = torch.cat((top_row, bottom_row), dim=2)
            elif args.transform == 'DCT_B':
                DTs = dct_b.dct_32_2d(blocks.type(torch.int32))
            else:
                DTs = dct.dct_2d(blocks)

            B_blocks, SC_blocks, energy = feature_extraction(args, DTs, steps, device)
            TC_blocks, TC2_blocks = temporal_feature_extraction(args, f, SC_blocks, energy, last_SC, last_energy)

            last_energy = energy[-2:]
            last_SC = SC_blocks[-2:]

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

        stream.close()

        if args.method == 'VCA':
            df = pd.DataFrame({'B': out_frames[0], 'E': out_frames[1], 'h': out_frames[2], 'h2': out_frames[3]})
        elif args.method == 'EVCA':
            df = pd.DataFrame({'B': out_frames[0], 'SC': out_frames[1], 'TC': out_frames[2], 'TC2': out_frames[3]})
        directory, file_name = os.path.split(args.csv)
        directory = './' if directory == '' else directory

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        df.to_csv(f'{directory}/{file_name[:-4]}_{args.method}.csv', index=False)
        if args.block_info == 0 and args.plot_info == 1:
            args.block_info = 1
            print("To plot features we set -bp 1.")

        if args.block_info:
            write_block_info(args, out_blocks[0], out_blocks[1], out_blocks[2], out_blocks[3])
        if args.plot_info:
            plot_block_info_EVCA(args)

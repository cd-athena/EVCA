import argparse
import os
import torch
import numpy as np
import pandas as pd
import torch_dct as dct
import libs.dct_butterfly_torch as dct_b
from pytorch_wavelets import DWTForward
from libs.frame_to_block import frame_to_block
from libs.feature_extraction import feature_extraction
from libs.write_block_info import write_block_info
from libs.plot_block_info_EVCA import plot_block_info_EVCA

def EVCA(args: argparse.Namespace, input_list ,device) -> None:
    width            = int(args.resolution.split('x')[0]) 
    height           = int(args.resolution.split('x')[1])

    for file in input_list:
        args.input = file
        blocks           = frame_to_block(args,device)
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

        B_blocks, SC_blocks, TC_blocks, TC2_blocks = feature_extraction(args,DTs,device)

        B_frame          = B_blocks.mean(dim=1)


        SC_frame         = SC_blocks.sum(dim=[1])/((width//args.block_size)*(height//args.block_size))
        TC_frame         = TC_blocks.sum(dim=[1])/((width//args.block_size)*(height//args.block_size))
        TC2_frame        = TC2_blocks.sum(dim=[1])/((width//args.block_size)*(height)//args.block_size)


        B_frame          = B_frame.cpu().numpy().ravel()
        SC_frame         = SC_frame.cpu().numpy().ravel()
        TC_frame         = np.insert(TC_frame.cpu().numpy().ravel(), 0, 0)
        TC2_frame        = np.insert(TC2_frame.cpu().numpy().ravel(), 0, 0)
        if len(np.arange(0,args.frames,args.sample_rate))>1:
            TC2_frame         = np.insert(TC2_frame, 0, 0)
        if args.method=='VCA':
           df = pd.DataFrame({'B':B_frame, 'E': SC_frame, 'h': TC_frame, 'h2': TC2_frame})
        elif args.method=='EVCA':
           df = pd.DataFrame({'B':B_frame, 'SC': SC_frame, 'TC': TC_frame, 'TC2': TC2_frame})
        directory, file_name = os.path.split(args.csv)
        directory = './' if directory == '' else directory

        if not os.path.exists(directory):
           os.makedirs(directory, exist_ok=True)
        df.to_csv(f'{directory}/{file_name[:-4]}_{args.method}.csv', index=False)
        if args.block_info==0 and args.plot_info==1:
            args.block_info = 1
            print("To plot features we set -bp 1.")

        if args.block_info:
            write_block_info(args,B_blocks,SC_blocks, TC_blocks,TC2_blocks)
        if args.plot_info:
            plot_block_info_EVCA(args)
        
 
    
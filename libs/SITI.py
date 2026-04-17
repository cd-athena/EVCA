import os

import numpy as np
import pandas as pd
import torch

from libs.edge_detection import edge_detection
from libs.plot_info_SITI import plot_info_SITI
from libs.video_to_frame import video_to_frame


def SITI(args, input_list, device):
    for file in input_list:
        args.input = file
        frames = video_to_frame(args, device)
        edge_frames = edge_detection(args, frames, device)

        batch_size = 64
        num_frames = frames.shape[0]
        SI_list = []
        TI_list = []
        TI_2_list = []

        for i in range(0, num_frames, batch_size):
            end_idx = min(i + batch_size, num_frames)
            batch_edges = edge_frames[i:end_idx]
            SI_batch = torch.std(batch_edges, dim=[1, 2]).cpu().numpy().ravel()
            SI_list.append(SI_batch)
            del batch_edges

        for i in range(0, num_frames - 1, batch_size):
            end_idx = min(i + batch_size, num_frames - 1)
            batch_frames = frames[i:end_idx+1]
            Mn = batch_frames[1:] - batch_frames[:-1]
            TI_batch = torch.std(Mn, dim=[1, 2]).cpu().numpy().ravel()
            TI_list.append(TI_batch)
            del Mn, batch_frames

        for i in range(0, num_frames - 2, batch_size):
            end_idx = min(i + batch_size, num_frames - 2)
            batch_frames = frames[i:end_idx+2]
            Mn_2 = batch_frames[2:] - batch_frames[:-2]
            TI_2_batch = torch.std(Mn_2, dim=[1, 2]).cpu().numpy().ravel()
            TI_2_list.append(TI_2_batch)
            del Mn_2, batch_frames

        SI = np.concatenate(SI_list)
        TI = np.concatenate(TI_list)
        TI_2 = np.concatenate(TI_2_list)

        TI = np.insert(TI, 0, 0)
        TI_2 = np.insert(TI_2, 0, 0)
        TI_2 = np.insert(TI_2, 0, 0)

        df = pd.DataFrame({'SI': SI, 'TI': TI, 'TI2': TI_2})
        directory, file_name = os.path.split(args.csv)
        directory = './' if directory == '' else directory
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        df.to_csv(f'{directory}/{file_name[:-4]}_SITI.csv', index=False)
        if args.block_info:
            print('block information is not available for SITI method.')
        if args.plot_info:
            plot_info_SITI(args, frames, edge_frames, frames[1:] - frames[:-1], frames[2:] - frames[:-2])
        del frames
        return edge_frames

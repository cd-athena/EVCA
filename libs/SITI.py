import os
import torch 
import numpy as np
import pandas as pd
from libs.video_to_frame import video_to_frame
from libs.edge_detection import edge_detection
from libs.plot_info_SITI import plot_info_SITI
def SITI(args,input_list,device):
    for file in input_list:
        args.input = file
        frames      = video_to_frame(args,device)
        edge_frames = edge_detection(args,frames, device)
        Mn          = frames[1:]-frames[:-1]
        Mn_2        = frames[2:]-frames[:-2]
        SI          = torch.std(edge_frames,dim=[1,2]).cpu().numpy().ravel()
        TI          = torch.std(Mn,dim=[1,2])
        TI_2        = torch.std(Mn_2,dim=[1,2])
        TI          = np.insert(TI.cpu().numpy().ravel(), 0, 0)
        TI_2        = np.insert(TI_2.cpu().numpy().ravel(), 0, 0)
        TI_2        = np.insert(TI_2, 0, 0)
        df = pd.DataFrame({'SI': SI, 'TI': TI, 'TI-2': TI_2})
        directory, file_name = os.path.split(args.csv)
        directory = './' if directory == '' else directory
        if not os.path.exists(directory):
           os.makedirs(directory, exist_ok=True)
        df.to_csv(f'{directory}/{file_name[:-4]}_SITI.csv', index=False)
        if args.block_info:
            print('block information is not available for SITI method.')
        if args.plot_info:
            plot_info_SITI(args,frames,edge_frames,Mn,Mn_2)
        return edge_frames
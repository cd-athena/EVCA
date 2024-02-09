import argparse
import torch
import numpy as np
from libs.edge_detection import edge_detection

def frame_to_edge(args: argparse.Namespace, device) -> torch.Tensor:
    stream          = open(args.input,  'rb')
    width           = int(args.resolution.split('x')[0]) 
    height          = int(args.resolution.split('x')[1])
    frms            = np.arange(0,args.frames,args.sample_rate)  
    frames          = []
    
    for frame in frms:
        if args.pix_fmt ==  'yuv420':
            stream.seek(frame * width * height * 3//2)
        elif args.pix_fmt == 'yuv444':
            stream.seek(frame * width * height * 3)
        Y = np.fromfile(stream, dtype=np.uint8, count=width * height).reshape(height, width)
        Y = torch.from_numpy(Y[:height//args.block_size*args.block_size,:]).to(device).float()
        frames.append(Y)
    frames  = torch.cat(frames, dim=0)
    frames  = frames.view(len(np.arange(0,args.frames,args.sample_rate)),height//args.block_size*args.block_size,width//args.block_size*args.block_size)
    edge_frames = edge_detection(frames, device)
    return edge_frames

import argparse
import torch
import numpy as np


def frame_to_block(args: argparse.Namespace, stream, start, end, device) -> torch.Tensor:

    width           = int(args.resolution.split('x')[0])
    height          = int(args.resolution.split('x')[1])
    frames          = np.arange(start, end, args.sample_rate)
    blocks          = []

    for frame in frames:
        if args.pix_fmt ==  'yuv420':
            stream.seek(frame * width * height * 3//2)
        elif args.pix_fmt == 'yuv444':
            stream.seek(frame * width * height * 3)
        Y = np.fromfile(stream, dtype=np.uint8, count=width * height).reshape(height, width)
        Y = torch.from_numpy(Y[:height//args.block_size*args.block_size,:width//args.block_size*args.block_size]).to(device)
        b = Y.unfold(0, args.block_size, args.block_size).unfold(1, args.block_size, args.block_size)
        b = b.contiguous().view(-1, args.block_size, args.block_size)
        blocks.append(b)
    blocks  = torch.cat(blocks, dim=0)
    return blocks

import numpy as np
import torch
from pathlib import Path



def video_to_frame(args, device):
    width = int(args.resolution.split('x')[0])
    height = int(args.resolution.split('x')[1])

    pix_size = 1.5
    if args.pix_fmt == 'yuv420':
        pix_size = 1.5
    elif args.pix_fmt == 'yuv444':
        pix_size = 3

    nframes = int(Path(args.input).stat().st_size // (width * height * pix_size))
    frms = np.arange(0, nframes, args.sample_rate)
    stream = open(args.input, 'rb')

    frames = []
    for frame in frms:
        stream.seek(int(frame * width * height * pix_size))
        Y = np.fromfile(stream, dtype=np.uint8, count=width * height).reshape(height, width)
        Y = torch.from_numpy(Y[:height // args.block_size * args.block_size, :]).to(device).float()
        frames.append(Y)
    frames = torch.cat(frames, dim=0)
    frames = frames.view(len(np.arange(0, nframes, args.sample_rate)), height // args.block_size * args.block_size,
                         width // args.block_size * args.block_size)
    return frames

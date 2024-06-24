import argparse
import torch
import numpy as np
from libs.weight_dct import weight_dct


def feature_extraction(args: argparse.Namespace, DCTs, nframes, device):
    width            = int(args.resolution.split('x')[0]) 
    height           = int(args.resolution.split('x')[1])

    ######## Brightness
    B_blocks         = DCTs.view(nframes,(width//args.block_size)*(height//args.block_size),args.block_size,args.block_size)
    B_blocks         = B_blocks[:,:,0,0]/(width*height)
    
    
    ######## Energy of Blocks
    weights_dct      = weight_dct(args,device)
    energy           = torch.abs(DCTs*weights_dct.unsqueeze(0))
    energy           = energy.view(len(np.arange(0,nframes,args.sample_rate)),(width//args.block_size)*(height//args.block_size),args.block_size,args.block_size)
    SC_blocks        = energy.mean(dim=[2, 3])/(args.block_size*args.block_size)

    return B_blocks, SC_blocks, energy


def temporal_feature_extraction(args: argparse.Namespace, start_frame, SC_blocks, energy, last_SC, last_energy):
    width = int(args.resolution.split('x')[0])
    height = int(args.resolution.split('x')[1])

    spatial_block = torch.cat((last_SC, SC_blocks), 0)
    energy_block = torch.cat((last_energy, energy), 0)

    ######## Temporal Complexity of Blocks
    if args.method == 'VCA':
        TC_blocks = torch.abs(spatial_block[1:] - spatial_block[:-1])
        TC2_blocks = torch.abs(spatial_block[2:] - spatial_block[:-2])
    elif args.method == 'EVCA':
        h_evca = torch.abs(energy_block[1+int(bool(start_frame)):] - energy_block[int(bool(start_frame)):-1])
        h2_evca = torch.abs(energy_block[2:] - energy_block[:-2])
        TC_blocks = h_evca.mean(dim=[2, 3]) / (args.block_size * args.block_size)
        TC2_blocks = h2_evca.mean(dim=[2, 3]) / (args.block_size * args.block_size)

    return TC_blocks, TC2_blocks
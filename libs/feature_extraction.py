import argparse

import numpy as np
import torch

from libs.weight_dct import weight_dct


def feature_extraction(args: argparse.Namespace, DCTs, nframes, device):
    width = int(args.resolution.split('x')[0])
    height = int(args.resolution.split('x')[1])

    ######## Brightness
    B_blocks = DCTs.view(nframes, (width // args.block_size) * (height // args.block_size), args.block_size,
                         args.block_size)
    B_blocks = B_blocks[:, :, 0, 0] / (width * height)

    ######## Energy of Blocks
    weights_dct = weight_dct(args, device)
    energy = torch.abs(DCTs * weights_dct.unsqueeze(0))
    energy = energy.view(nframes,
                         (width // args.block_size) * (height // args.block_size), 
                         args.block_size, 
                         args.block_size)
    SC_blocks = energy.mean(dim=[2, 3]) / (args.block_size * args.block_size)

    return B_blocks, SC_blocks, energy


def temporal_feature_extraction(args: argparse.Namespace, start_frame, SC_blocks, energy, last_SC, last_energy):
    width = int(args.resolution.split('x')[0])
    height = int(args.resolution.split('x')[1])

    spatial_block = torch.cat((last_SC, SC_blocks), 0)
    energy_block = torch.cat((last_energy, energy), 0)

    ######## Temporal Complexity of Blocks
    if args.method == 'VCA':
        TC_blocks = torch.abs(spatial_block[1 + int(bool(start_frame)):] - spatial_block[int(bool(start_frame)):-1])
        TC2_blocks = torch.abs(spatial_block[2:] - spatial_block[:-2])
    elif args.method == 'EVCA':
        h_evca = torch.abs(energy_block[1 + int(bool(start_frame)):] - energy_block[int(bool(start_frame)):-1])
        h2_evca = torch.abs(energy_block[2:] - energy_block[:-2])
        TC_blocks = h_evca.mean(dim=[2, 3]) / (args.block_size * args.block_size)
        TC2_blocks = h2_evca.mean(dim=[2, 3]) / (args.block_size * args.block_size)

    return TC_blocks, TC2_blocks

def chroma_energy_extraction(args: argparse.Namespace,
                             U_DCTs: torch.Tensor,
                             V_DCTs: torch.Tensor,
                             nframes: int,
                             chroma_weight_dct: torch.Tensor) -> torch.Tensor:
    """
    Computes the Chroma DCT Energy (E_c) for U and V channels and combines them.
    E_c = (E_u + E_v) / 2
    """
    width, height = map(int, args.resolution.split('x'))
    if args.pix_fmt == 'yuv420':
        uv_w, uv_h = width // 2, height // 2
        cb_size = args.block_size // 2
    elif args.pix_fmt == 'yuv444':
        uv_w, uv_h = width, height
        cb_size = args.block_size
    
    num_blocks_per_frame = (uv_w // cb_size) * (uv_h // cb_size)
    
    # calc equivalents of AC coefficients.
    # unsqueeze broadcasts the weights across all blocks instantly
    energy_u = torch.abs(U_DCTs * chroma_weight_dct.unsqueeze(0))
    energy_v = torch.abs(V_DCTs * chroma_weight_dct.unsqueeze(0))
    
    # reshape back to distinct frames and blocks
    energy_u = energy_u.view(nframes, num_blocks_per_frame, cb_size, cb_size)
    energy_v = energy_v.view(nframes, num_blocks_per_frame, cb_size, cb_size)
    
    # calc mean energy per block normalized by block area
    sc_blocks_u = energy_u.mean(dim=[2, 3]) / (cb_size * cb_size)
    sc_blocks_v = energy_v.mean(dim=[2, 3]) / (cb_size * cb_size)
    
    # combine U and V spatial complexities into single Chroma Spatial metric (SC_c)
    SC_chroma_blocks = (sc_blocks_u + sc_blocks_v) * 0.5
    
    return SC_chroma_blocks
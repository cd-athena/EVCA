import argparse
import torch
import numpy as np

def edge_detection(args: argparse.Namespace, frames, device) -> torch.Tensor:
    if args.filter == 'sobel':
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([
            [-1, 0, 1], 
            [-2, 0, 2], 
            [-1, 0, 1]],
            dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1], 
            [0,   0,  0], 
            [1,   2,  1]], 
            dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        # Applying Sobel filters
        edge_x = torch.nn.functional.conv2d(frames.unsqueeze(1), sobel_x, padding=1)
        edge_y = torch.nn.functional.conv2d(frames.unsqueeze(1), sobel_y, padding=1)

        # Calculating the magnitude of edges
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2).squeeze(1)
        
    elif args.filter == 'canny':
        matrix_data = [
            [0.07511361, 0.1238414, 0.07511361],
            [0.1238414, 0.20417996, 0.1238414],
            [0.07511361, 0.1238414, 0.07511361]
               ]
        gaussian_2D = torch.tensor(matrix_data, dtype=torch.float32)
        print(gaussian_2D)

    return edges
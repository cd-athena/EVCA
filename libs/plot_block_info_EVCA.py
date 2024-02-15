import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_block_info_EVCA(args):
    if not os.path.exists('png/'):
        os.makedirs('png/', exist_ok=True)

    stream          = open(args.input,  'rb')
    width           = int(args.resolution.split('x')[0]) 
    height          = int(args.resolution.split('x')[1])
    frames          = np.arange(0,args.frames,args.sample_rate)  
    blocks          = []
    
    for frame in frames:
        fig, axes = plt.subplots(1, 4, figsize=(12, 5))
        
        if args.pix_fmt ==  'yuv420':
            stream.seek(frame * width * height * 3//2)
        elif args.pix_fmt == 'yuv444':
            stream.seek(frame * width * height * 3)
        Y = np.fromfile(stream, dtype=np.uint8, count=width * height).reshape(height, width)
        image1 = Y
        
        df = pd.read_csv(f'{args.csv[:-4]}_B_blocks.csv')
        B  = df[f'frame_{frame:03d}'].values.reshape(int(height//args.block_size),int( width//args.block_size))
        image2 = B

        df = pd.read_csv(f'{args.csv[:-4]}_SC_blocks.csv')
        SC  = df[f'frame_{frame:03d}'].values.reshape(int(height//args.block_size),int( width//args.block_size))
        image3 = SC

        df = pd.read_csv(f'{args.csv[:-4]}_TC_blocks.csv')
        TC  = df[f'frame_{frame:03d}'].values.reshape(int(height//args.block_size),int( width//args.block_size))
        image4 = TC

        
        axes[0].imshow(image1, cmap='gray')
        axes[0].set_title(f'Orginal Frame')
        axes[0].axis('off')  # Turn off axis
        
        axes[1].imshow(image2, cmap='gray')
        axes[1].set_title('Brightness')
        axes[1].axis('off')  # Turn off axis
        
        axes[2].imshow(image3, cmap='gray')
        axes[2].set_title('Spatial Complexity')
        axes[2].axis('off')  # Turn off axis

        axes[3].imshow(image4, cmap='gray')
        axes[3].set_title('Temporal Complexity')
        axes[3].axis('off')  # Turn off axis
     
        plt.savefig(f'png/{args.method}_frame_{frame:03d}.png', bbox_inches='tight',dpi=args.dpi)
        plt.close()

        

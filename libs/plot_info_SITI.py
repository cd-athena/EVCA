import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def plot_info_SITI(args,frames,SI,TI,TI_2):
    if not os.path.exists('png/'):
        os.makedirs('png/', exist_ok=True)
    
    for frame in range(frames.shape[0]):
        fig, axes = plt.subplots(1, 4, figsize=(12, 5)) 
        image1 = frames[frame].cpu()
        axes[0].imshow(image1, cmap='gray')
        axes[0].set_title(f'Orginal Frame')
        axes[0].axis('off')  # Turn off axis
        
        image2 = SI[frame].cpu()
        axes[1].imshow(image2, cmap='gray')
        axes[1].set_title('Spatial Compelxity (SI)')
        axes[1].axis('off')  # Turn off axis

        if frame == 0: 
            image3 = TI[frame].cpu()*0
        else:
            image3 = TI[frame-1].cpu()
        axes[2].imshow(image3, cmap='gray')
        axes[2].set_title('Temporal Complexity (TI)')
        axes[2].axis('off')  # Turn off axis

        if frame == 0 or frame ==1:
            image4 = TI_2[0].cpu()*0
        else:
            image4 = TI_2[frame-2].cpu()
        axes[3].imshow(image4, cmap='gray')
        axes[3].set_title('Temporal Complexity ($TI_{-2}$)')
        axes[3].axis('off')  # Turn off axis
    
     
        plt.savefig(f'png/SITI_frame_{frame:03d}.png', bbox_inches='tight',dpi=args.dpi)
        plt.close()

        

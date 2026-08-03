import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_block_info_EVCA(args, number_of_frames):
    if not os.path.exists('png/'):
        os.makedirs('png/', exist_ok=True)

    stream = open(args.input, 'rb')
    width = int(args.resolution.split('x')[0])
    height = int(args.resolution.split('x')[1])
    n_frames = args.frames if args.frames != 0 else number_of_frames
    frames = np.arange(0, n_frames, args.sample_rate)
    grid_h = int(height // args.block_size)
    grid_w = int(width // args.block_size)

    # Load block CSV data if files exist
    csv_base = args.csv[:-4]
    
    df_B = pd.read_csv(f'{csv_base}_B_blocks.csv') if os.path.exists(f'{csv_base}_B_blocks.csv') else None
    df_SC = pd.read_csv(f'{csv_base}_SC_blocks.csv') if os.path.exists(f'{csv_base}_SC_blocks.csv') else None
    df_TC = pd.read_csv(f'{csv_base}_TC_blocks.csv') if os.path.exists(f'{csv_base}_TC_blocks.csv') else None
    
    df_SC_u = pd.read_csv(f'{csv_base}_SC_u_blocks.csv') if args.chroma_complexity and os.path.exists(f'{csv_base}_SC_u_blocks.csv') else None
    df_SC_v = pd.read_csv(f'{csv_base}_SC_v_blocks.csv') if args.chroma_complexity and os.path.exists(f'{csv_base}_SC_v_blocks.csv') else None
    
    is_me = getattr(args, 'motion_estimation', False)
    df_SAD = pd.read_csv(f'{csv_base}_SAD_blocks.csv') if is_me and os.path.exists(f'{csv_base}_SAD_blocks.csv') else None
    df_MV = pd.read_csv(f'{csv_base}_MV_blocks.csv') if is_me and os.path.exists(f'{csv_base}_MV_blocks.csv') else None
    df_TCMC = pd.read_csv(f'{csv_base}_TCMC_blocks.csv') if is_me and getattr(args, 'profile', 'fast') == 'full' and os.path.exists(f'{csv_base}_TCMC_blocks.csv') else None

    for frame in frames:
        col_name = f'frame_{frame:03d}'
        
        if args.pix_fmt == 'yuv420':
            stream.seek(int(frame) * int(width) * int(height) * 3 // 2)
        elif args.pix_fmt == 'yuv444':
            stream.seek(int(frame) * int(width) * int(height) * 3)
        Y = np.fromfile(stream, dtype=np.uint8, count=width * height).reshape(height, width)
        
        plot_items = [(Y, 'Original Frame')]
        
        if df_B is not None and col_name in df_B.columns:
            plot_items.append((df_B[col_name].values.reshape(grid_h, grid_w), 'Brightness'))
        if df_SC is not None and col_name in df_SC.columns:
            plot_items.append((df_SC[col_name].values.reshape(grid_h, grid_w), 'Spatial Complexity'))
        if df_TC is not None and col_name in df_TC.columns:
            plot_items.append((df_TC[col_name].values.reshape(grid_h, grid_w), 'Temporal Complexity'))
            
        if df_SC_u is not None and col_name in df_SC_u.columns:
            plot_items.append((df_SC_u[col_name].values.reshape(grid_h, grid_w), 'Chroma SC (U)'))
        if df_SC_v is not None and col_name in df_SC_v.columns:
            plot_items.append((df_SC_v[col_name].values.reshape(grid_h, grid_w), 'Chroma SC (V)'))

        if df_SAD is not None and col_name in df_SAD.columns:
            plot_items.append((df_SAD[col_name].values.reshape(grid_h, grid_w), 'TC_SAD (Block-level SAD prediction error)'))
        if df_MV is not None and col_name in df_MV.columns:
            plot_items.append((df_MV[col_name].values.reshape(grid_h, grid_w), 'MVC (Block-level motion vector magnitudes)'))
        if df_TCMC is not None and col_name in df_TCMC.columns:
            plot_items.append((df_TCMC[col_name].values.reshape(grid_h, grid_w), 'TC_MC (Motion-compensated residual complexity)'))

        num_plots = len(plot_items)
        if num_plots <= 5:
            nrows, ncols = 1, num_plots
        else:
            nrows = 2
            ncols = (num_plots + 1) // 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
        axes_flat = np.array(axes).reshape(-1) if num_plots > 1 else np.array([axes])

        for idx, (img, title) in enumerate(plot_items):
            axes_flat[idx].imshow(img, cmap='gray')
            axes_flat[idx].set_title(title)
            axes_flat[idx].axis('off')

        for idx in range(num_plots, nrows * ncols):
            axes_flat[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'png/{args.method}_frame_{frame:03d}.png', bbox_inches='tight', dpi=args.dpi)
        plt.close(fig)

    stream.close()


import os
import numpy as np
import pandas as pd

def write_block_info(args,B_blocks,SC_blocks,TC_blocks,TC2_blocks):
    directory, file_name = os.path.split(args.csv)
    directory = './' if directory == '' else directory
 
    B_blocks         = B_blocks.view(len(np.arange(0,args.frames,args.sample_rate)),-1)
    B_blocks         = B_blocks.cpu().numpy()
    df_B_blocks      = pd.DataFrame(columns=[f'frame_{i:03d}' for i in range(0, len(np.arange(0,args.frames,args.sample_rate)))])
    for i in range(len(np.arange(0,args.frames,args.sample_rate))):
        df_B_blocks[f'frame_{i:03d}'] = B_blocks[i, :]
    df_B_blocks.to_csv(f'{directory}/{file_name[:-4]}_B_blocks.csv', index=False)
 
    SC_blocks         = SC_blocks.view(len(np.arange(0,args.frames,args.sample_rate)),-1)
    
    SC_blocks         = SC_blocks.cpu().numpy()
    df_SC_blocks      = pd.DataFrame(columns=[f'frame_{i:03d}' for i in range(0, len(np.arange(0,args.frames,args.sample_rate)))])

    for i in range(len(np.arange(0,args.frames,args.sample_rate))):  
        df_SC_blocks[f'frame_{i:03d}'] = SC_blocks[i, :]
    
    df_SC_blocks.to_csv(f'{directory}/{file_name[:-4]}_SC_blocks.csv', index=False)
   
    
    TC_blocks         = TC_blocks.view(len(np.arange(0,args.frames,args.sample_rate))-1,-1)
    
    TC_blocks         = TC_blocks.cpu().numpy()

    df_TC_blocks      = pd.DataFrame(columns=[f'frame_{i:03d}' for i in range(0, len(np.arange(0,args.frames,args.sample_rate)))])

    for i in range(0,len(np.arange(0,args.frames,args.sample_rate))):  
        if i==0:
            df_TC_blocks[f'frame_{i:03d}'] = np.zeros(TC_blocks.shape[1])
        else:
            df_TC_blocks[f'frame_{i:03d}'] = TC_blocks[i-1, :]
    
    df_TC_blocks.to_csv(f'{directory}/{file_name[:-4]}_TC_blocks.csv', index=False)

    TC2_blocks         = TC2_blocks.view(len(np.arange(0,args.frames,args.sample_rate))-2,-1)
    
    TC2_blocks         = TC2_blocks.cpu().numpy()
    df_TC2_blocks      = pd.DataFrame(columns=[f'frame_{i:03d}' for i in range(0, len(np.arange(0,args.frames,args.sample_rate)))])

    for i in range(0,len(np.arange(0,args.frames,args.sample_rate))):  
        if i<2:
            df_TC2_blocks[f'frame_{i:03d}'] = np.zeros(TC2_blocks.shape[1])
        else:
            df_TC2_blocks[f'frame_{i:03d}'] = TC_blocks[i-2, :]
    
    df_TC_blocks.to_csv(f'{directory}/{file_name[:-4]}_TC2_blocks.csv', index=False)
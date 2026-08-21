import os
import torch
import numpy as np
import pandas as pd


def write_block_info(args, B_blocks, SC_blocks, TC_blocks, TC2_blocks, number_of_frames, SC_u=None, SC_v=None, sad_blocks=None, mv_blocks=None, tcmc_blocks=None):
    directory, file_name = os.path.split(args.csv)
    directory = './csv' if directory == '' else directory

    n_frames = args.frames if args.frames != 0 else number_of_frames
    B_blocks = torch.cat(B_blocks, dim=0)
    B_blocks = B_blocks.view(len(np.arange(0, n_frames, args.sample_rate)), -1)
    B_blocks = B_blocks.cpu().numpy()
    df_B_blocks = pd.DataFrame(
        columns=[f'frame_{i:03d}' for i in range(0, len(np.arange(0, n_frames, args.sample_rate)))])
    for i in range(len(np.arange(0, n_frames, args.sample_rate))):
        df_B_blocks[f'frame_{i:03d}'] = B_blocks[i, :]
    df_B_blocks.to_csv(f'{directory}/{file_name[:-4]}_B_blocks.csv', index=False)

    # ----------------- SC_blocks -----------------
    SC_blocks = torch.cat(SC_blocks, dim=0)
    SC_blocks = SC_blocks.view(len(np.arange(0, n_frames, args.sample_rate)), -1)

    SC_blocks = SC_blocks.cpu().numpy()
    df_SC_blocks = pd.DataFrame(
        columns=[f'frame_{i:03d}' for i in range(0, len(np.arange(0, n_frames, args.sample_rate)))])

    for i in range(len(np.arange(0, n_frames, args.sample_rate))):
        df_SC_blocks[f'frame_{i:03d}'] = SC_blocks[i, :]

    df_SC_blocks.to_csv(f'{directory}/{file_name[:-4]}_SC_blocks.csv', index=False)

    # ----------------- TC_blocks -----------------
    TC_blocks = torch.cat(TC_blocks, dim=0)
    TC_blocks = TC_blocks.view(len(np.arange(0, n_frames, args.sample_rate)) - 1, -1)

    TC_blocks = TC_blocks.cpu().numpy()

    df_TC_blocks = pd.DataFrame(
        columns=[f'frame_{i:03d}' for i in range(0, len(np.arange(0, n_frames, args.sample_rate)))])

    for i in range(0, len(np.arange(0, n_frames, args.sample_rate))):
        if i == 0:
            df_TC_blocks[f'frame_{i:03d}'] = np.zeros(TC_blocks.shape[1])
        else:
            df_TC_blocks[f'frame_{i:03d}'] = TC_blocks[i - 1, :]

    df_TC_blocks.to_csv(f'{directory}/{file_name[:-4]}_TC_blocks.csv', index=False)

    # ----------------- TC2_blocks -----------------
    TC2_blocks = torch.cat(TC2_blocks, dim=0)
    TC2_blocks = TC2_blocks.view(len(np.arange(0, n_frames, args.sample_rate)) - 2, -1)

    TC2_blocks = TC2_blocks.cpu().numpy()
    df_TC2_blocks = pd.DataFrame(
        columns=[f'frame_{i:03d}' for i in range(0, len(np.arange(0, n_frames, args.sample_rate)))])

    for i in range(0, len(np.arange(0, n_frames, args.sample_rate))):
        if i < 2:
            df_TC2_blocks[f'frame_{i:03d}'] = np.zeros(TC2_blocks.shape[1])
        else:
            df_TC2_blocks[f'frame_{i:03d}'] = TC2_blocks[i - 2, :]

    df_TC2_blocks.to_csv(f'{directory}/{file_name[:-4]}_TC2_blocks.csv', index=False)

    # ----------------- SC_u blocks -----------------
    if SC_u is not None:
        SC_u = torch.cat(SC_u, dim=0)
        SC_u = SC_u.view(len(np.arange(0, n_frames, args.sample_rate)), -1)
        
        SC_u = SC_u.cpu().numpy()
        df_SC_u_blocks = pd.DataFrame(
            columns=[f'frame_{i:03d}' for i in range(0, len(np.arange(0, n_frames, args.sample_rate)))])
        
        for i in range(len(np.arange(0, n_frames, args.sample_rate))):
            df_SC_u_blocks[f'frame_{i:03d}'] = SC_u[i, :]
            
        df_SC_u_blocks.to_csv(f'{directory}/{file_name[:-4]}_SC_u_blocks.csv', index=False)
        
    # ----------------- SC_v blocks -----------------
    if SC_v is not None:
        SC_v = torch.cat(SC_v, dim=0)
        SC_v = SC_v.view(len(np.arange(0, n_frames, args.sample_rate)), -1)
        
        SC_v = SC_v.cpu().numpy()
        df_SC_v_blocks = pd.DataFrame(
            columns=[f'frame_{i:03d}' for i in range(0, len(np.arange(0, n_frames, args.sample_rate)))])
        
        for i in range(len(np.arange(0, n_frames, args.sample_rate))):
            df_SC_v_blocks[f'frame_{i:03d}'] = SC_v[i, :]
            
        df_SC_v_blocks.to_csv(f'{directory}/{file_name[:-4]}_SC_v_blocks.csv', index=False)

    # ----------------- SAD_blocks -----------------
    if sad_blocks is not None and len(sad_blocks) > 0:
        sad_tensor = torch.cat(sad_blocks, dim=0)
        sad_tensor = sad_tensor.view(len(np.arange(0, n_frames, args.sample_rate)) - 1, -1).cpu().numpy()
        df_sad = pd.DataFrame(columns=[f'frame_{i:03d}' for i in range(0, len(np.arange(0, n_frames, args.sample_rate)))])
        for i in range(0, len(np.arange(0, n_frames, args.sample_rate))):
            if i == 0:
                df_sad[f'frame_{i:03d}'] = np.zeros(sad_tensor.shape[1])
            else:
                df_sad[f'frame_{i:03d}'] = sad_tensor[i - 1, :]
        df_sad.to_csv(f'{directory}/{file_name[:-4]}_SAD_blocks.csv', index=False)

    # ----------------- MV_blocks -----------------
    if mv_blocks is not None and len(mv_blocks) > 0:
        mv_tensor = torch.cat(mv_blocks, dim=0)
        mv_tensor = mv_tensor.view(len(np.arange(0, n_frames, args.sample_rate)) - 1, -1).cpu().numpy()
        df_mv = pd.DataFrame(columns=[f'frame_{i:03d}' for i in range(0, len(np.arange(0, n_frames, args.sample_rate)))])
        for i in range(0, len(np.arange(0, n_frames, args.sample_rate))):
            if i == 0:
                df_mv[f'frame_{i:03d}'] = np.zeros(mv_tensor.shape[1])
            else:
                df_mv[f'frame_{i:03d}'] = mv_tensor[i - 1, :]
        df_mv.to_csv(f'{directory}/{file_name[:-4]}_MV_blocks.csv', index=False)

    # ----------------- TCMC_blocks -----------------
    if tcmc_blocks is not None and len(tcmc_blocks) > 0:
        tcmc_tensor = torch.cat(tcmc_blocks, dim=0)
        tcmc_tensor = tcmc_tensor.view(len(np.arange(0, n_frames, args.sample_rate)) - 1, -1).cpu().numpy()
        df_tcmc = pd.DataFrame(columns=[f'frame_{i:03d}' for i in range(0, len(np.arange(0, n_frames, args.sample_rate)))])
        for i in range(0, len(np.arange(0, n_frames, args.sample_rate))):
            if i == 0:
                df_tcmc[f'frame_{i:03d}'] = np.zeros(tcmc_tensor.shape[1])
            else:
                df_tcmc[f'frame_{i:03d}'] = tcmc_tensor[i - 1, :]
        df_tcmc.to_csv(f'{directory}/{file_name[:-4]}_TCMC_blocks.csv', index=False)
import pandas as pd
import os
import argparse
from pathlib import Path

def export_features_to_csv(args: argparse.Namespace, file: str, out_frames: list, out_frames_chroma: list) -> str:
    """
    Exports the computed features into a CSV file based on the selected method.
    Returns the path to the saved CSV file.
    """
    if args.method == 'VCA':
        data = {'B': out_frames[0], 'E': out_frames[1], 'h': out_frames[2], 'h2': out_frames[3]}
    elif args.method == 'EVCA':
        if args.chroma_complexity:
            data = {'B': out_frames[0], 'SC': out_frames[1], 'TC': out_frames[2], 'TC2': out_frames[3], 'SC_c': out_frames_chroma}
        else:
            data = {'B': out_frames[0], 'SC': out_frames[1], 'TC': out_frames[2], 'TC2': out_frames[3]}
            
    if args.colorfulness:
        data['Colorfulness'] = out_frames[4]
        
    df = pd.DataFrame(data)
    
    directory, file_name = os.path.split(args.csv)
    directory = './' if directory == '' else directory

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    if args.dir:
        final_csv_path = f'{directory}/{file_name[:-4]}_{args.method}_{Path(file).name[:-4]}.csv'
    else:
        final_csv_path = args.csv
        
    df.to_csv(final_csv_path, index=False)
    return final_csv_path

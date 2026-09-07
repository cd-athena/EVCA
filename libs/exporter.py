import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

_GIT_SHA = None


def get_git_sha() -> str:
    """Best-effort git SHA of the repo this module lives in (cached)."""
    global _GIT_SHA
    if _GIT_SHA is None:
        try:
            _GIT_SHA = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=Path(__file__).resolve().parent.parent,
                text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            _GIT_SHA = 'unknown'
    return _GIT_SHA


def write_provenance_sidecar(csv_path: str, args: argparse.Namespace, input_file: str) -> None:
    """Writes <csv>.meta.json with the git SHA and the full argument namespace."""
    meta = {
        'git_sha': get_git_sha(),
        'input_file': str(input_file),
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'args': {k: v for k, v in vars(args).items()},
    }
    with open(f'{csv_path}.meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)


def export_features_to_csv(args: argparse.Namespace, file: str, out_frames: list,
                           out_frames_u: list = None, out_frames_v: list = None,
                           out_mvc: list = None, out_tcsad: list = None,
                           out_tcmc: list = None,
                           out_meanmv: list = None,
                           out_intrafrac: list = None) -> str:
    """
    Exports the computed features into a CSV file based on the selected method.
    Returns the path to the saved CSV file.
    """
    if args.method == 'VCA':
        data = {'B': out_frames[0], 'E': out_frames[1], 'h': out_frames[2], 'h2': out_frames[3]}
    elif args.method == 'EVCA':
        if args.chroma_complexity and out_frames_u and out_frames_v:
            data = {'B': out_frames[0], 'SC': out_frames[1], 'TC': out_frames[2], 'TC2': out_frames[3],
                    'SC_u': out_frames_u, 'SC_v': out_frames_v}
        else:
            data = {'B': out_frames[0], 'SC': out_frames[1], 'TC': out_frames[2], 'TC2': out_frames[3]}

    if args.colorfulness:
        data['Colorfulness'] = out_frames[4]

    if getattr(args, 'motion_estimation', False):
        if out_mvc and out_tcsad:
            data['MVC'] = out_mvc
            data['TC_SAD'] = out_tcsad
        if out_tcmc:
            data['TC_MC'] = out_tcmc
        # ME diagnostics: mean MV magnitude, plus the intra-gate fire rate in the
        # full profile.
        if out_meanmv:
            data['mean_mv_mag'] = out_meanmv
        if out_intrafrac:
            data['intra_frac'] = out_intrafrac

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
    write_provenance_sidecar(final_csv_path, args, file)
    return final_csv_path

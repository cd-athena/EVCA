"""x265 ground-truth generation with per-frame bit accounting.

Two encodes per (sequence, QP):
  * All-Intra  (keyint=1)                 -> per-frame I bits, the SC ground truth
  * Low-Delay P (keyint=9999, bframes=0)  -> per-frame P bits, the TC ground truth

Encoded bitstreams are cached under `cache_dir` and keyed by (sequence, mode, QP,
frame count), so re-running the harness re-probes instead of re-encoding.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / 'sequences.json'


def ffmpeg_available() -> bool:
    """True when ffmpeg/ffprobe exist and ffmpeg has libx265."""
    if not (shutil.which('ffmpeg') and shutil.which('ffprobe')):
        return False
    try:
        out = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        return False
    return 'libx265' in out


def load_config(config_path: Path = DEFAULT_CONFIG, sequence_root: Optional[str] = None) -> Dict:
    """Loads sequences.json and resolves sequence paths.

    Root precedence: explicit argument > EVCA_SEQUENCE_ROOT env var > file value.
    Sequences whose file is missing are dropped and reported in `missing`; entries
    carrying `"enabled": false` are deliberately out of the corpus and reported in
    `disabled` instead, so an oversized sequence can stay documented without being
    mistaken for an accidentally absent one.
    """
    with open(config_path) as f:
        cfg = json.load(f)
    root = Path(sequence_root or os.environ.get('EVCA_SEQUENCE_ROOT')
                or cfg.get('sequence_root', '.'))
    resolved, missing, disabled = [], [], []
    for seq in cfg['sequences']:
        p = Path(seq['path'])
        full = p if p.is_absolute() else root / p
        seq = dict(seq, full_path=str(full))
        if not seq.get('enabled', True):
            disabled.append(seq)
        else:
            (resolved if full.exists() else missing).append(seq)
    cfg['sequences'] = resolved
    cfg['missing'] = missing
    cfg['disabled'] = disabled
    cfg['sequence_root'] = str(root)
    return cfg


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, check=True)


def probe_frames(mp4_path: Path) -> pd.DataFrame:
    """Per-frame (frame_idx, pict_type, bits) for an encoded bitstream."""
    res = _run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'frame=pict_type,pkt_size', '-of', 'csv=p=0',
                str(mp4_path)])
    rows = []
    for line in res.stdout.strip().split('\n'):
        if not line.strip():
            continue
        # Field order varies across ffprobe builds; identify tokens by content.
        parts = [p.strip() for p in line.split(',') if p.strip()]
        size = next((int(p) for p in parts if p.isdigit()), None)
        ptype = next((p.upper() for p in parts if p.upper() in ('I', 'P', 'B')), None)
        if size is not None:
            rows.append({'frame_idx': len(rows), 'pict_type': ptype, 'bits': size * 8})
    return pd.DataFrame(rows)


def encode(seq: Dict, qp: int, mode: str, n_frames: int, cache_dir: Path,
           force: bool = False, preset: str = 'medium') -> tuple:
    """Encodes (or reuses) one bitstream. Returns (path, encode_seconds_or_None)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_tag = 'all' if n_frames == 0 else str(n_frames)
    out = cache_dir / f"{seq['name']}_{mode}_qp{qp}_f{frame_tag}.mp4"
    if out.exists() and not force:
        return out, None

    if mode == 'ai':
        x265_params = f'keyint=1:no-open-gop=1:qp={qp}:log-level=none'
    elif mode == 'ldp':
        x265_params = f'keyint=9999:bframes=0:no-scenecut=1:qp={qp}:log-level=none'
    else:
        raise ValueError(f'unknown encode mode: {mode}')

    pix_fmt = 'yuv420p' if seq['pix_fmt'] == 'yuv420' else 'yuv444p'
    if seq.get('bit_depth', 8) > 8:
        pix_fmt = pix_fmt.replace('p', f"p{seq['bit_depth']}le")

    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', pix_fmt,
           '-video_size', seq['res'], '-framerate', str(seq['fps']),
           '-i', seq['full_path']]
    if n_frames:
        cmd += ['-frames:v', str(n_frames)]
    cmd += ['-c:v', 'libx265', '-preset', preset, '-x265-params', x265_params, str(out)]

    t0 = time.time()
    _run(cmd)
    return out, time.time() - t0


def build_ground_truth(sequences: List[Dict], qps: List[int], n_frames: int,
                       cache_dir: Path, force: bool = False, preset: str = 'medium',
                       verbose: bool = True) -> pd.DataFrame:
    """Per-frame ground truth for every (sequence, QP).

    Returns a long DataFrame with columns
    `seq_name, qp, frame_idx, pict_type, bits_ai, bits_ldp`, where `bits_ai` is the
    All-Intra frame size (SC ground truth) and `bits_ldp` the Low-Delay-P frame size
    (TC ground truth). Frame 0 of the LDP stream is the I-frame and is retained here;
    consumers exclude it.
    """
    records = []
    for seq in sequences:
        for qp in qps:
            if verbose:
                print(f"  [GT] {seq['name']} QP {qp}", flush=True)
            ai_path, t_ai = encode(seq, qp, 'ai', n_frames, cache_dir, force, preset)
            ldp_path, t_ldp = encode(seq, qp, 'ldp', n_frames, cache_dir, force, preset)
            df_ai = probe_frames(ai_path).rename(columns={'bits': 'bits_ai'})
            df_ldp = probe_frames(ldp_path).rename(columns={'bits': 'bits_ldp'})
            df = df_ldp.merge(df_ai[['frame_idx', 'bits_ai']], on='frame_idx', how='inner')
            df['seq_name'] = seq['name']
            df['qp'] = qp
            df['t_enc_ldp'] = t_ldp if t_ldp is not None else float('nan')
            records.append(df)
    if not records:
        return pd.DataFrame(columns=['seq_name', 'qp', 'frame_idx', 'pict_type',
                                     'bits_ai', 'bits_ldp', 't_enc_ldp'])
    out = pd.concat(records, ignore_index=True)
    return out[['seq_name', 'qp', 'frame_idx', 'pict_type', 'bits_ai', 'bits_ldp', 't_enc_ldp']]


def sequence_mean_table(df_frames: pd.DataFrame) -> pd.DataFrame:
    """Legacy sequence-mean ground truth (SC_gt from AI frames, TC_gt from P frames)."""
    rows = []
    for (seq_name, qp), g in df_frames.groupby(['seq_name', 'qp']):
        p_frames = g[g['pict_type'] == 'P']
        rows.append({
            'seq_name': seq_name,
            'qp': qp,
            'SC_gt': float(g['bits_ai'].mean()) if len(g) else float('nan'),
            'TC_gt': float(p_frames['bits_ldp'].mean()) if len(p_frames) else float('nan'),
            'T_enc': float(g['t_enc_ldp'].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values(['seq_name', 'qp']).reset_index(drop=True)

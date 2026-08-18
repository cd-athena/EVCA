import os
import re
import sys
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# ==============================================================================
# CONFIGURATION & TEST SEQUENCES
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent  # validation/
PROJECT_ROOT = SCRIPT_DIR.parent

EVCA_MAIN_PATH = str(PROJECT_ROOT / "main.py")
OUTPUT_DIR = SCRIPT_DIR / "benchmark_results"
TEMP_DIR = OUTPUT_DIR / "temp"

GT_CSV = OUTPUT_DIR / "ground_truth_results.csv"
EVCA_CSV = OUTPUT_DIR / "evca_feature_results.csv"
CORR_CSV = OUTPUT_DIR / "correlation_matrix.csv"

QPS = [22, 27, 32, 37]

TEST_SEQUENCES = [
    {
        "path": "/Users/albert/Desktop/athena/test_sequences/YachtRide_1920x1080_120fps_420_8bit_YUV.yuv",
        "res": "1920x1080",
        "fps": 120,
        "pix_fmt": "yuv420",
        "bit_depth": 8,
    },
    {
        "path": "/Users/albert/Desktop/athena/test_sequences/ReadySteadyGo_1920x1080_120fps_420_8bit_YUV.yuv",
        "res": "1920x1080",
        "fps": 120,
        "pix_fmt": "yuv420",
        "bit_depth": 8,
    },
    {
        "path": "/Users/albert/Desktop/athena/test_sequences/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv",
        "res": "1920x1080",
        "fps": 120,
        "pix_fmt": "yuv420",
        "bit_depth": 8,
    },
    {
        "path": "/Users/albert/Desktop/athena/test_sequences/foodmarket_1920x1080_60fps_420_8bit.yuv",
        "res": "1920x1080",
        "fps": 60,
        "pix_fmt": "yuv420",
        "bit_depth": 8,
    },
]

# ==============================================================================
# HELPER FUNCTIONS & PROBERS
# ==============================================================================
def run_cmd(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

def parse_ai_frame_sizes(mp4_path: Path) -> float:
    """Probes an All-Intra MP4 file to compute SC_gt (bits per frame)."""
    if not mp4_path.exists():
        print(f"Warning: File {mp4_path} not found.")
        return 0.0
    cmd_probe_ai = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pkt_size", "-of", "csv=p=0", str(mp4_path)
    ]
    res = run_cmd(cmd_probe_ai)
    ai_sizes = [int(line.split(",")[0].strip()) for line in res.stdout.strip().split("\n") if line.strip()]
    return float(np.mean(ai_sizes)) * 8.0 if ai_sizes else 0.0

def reparse_ldp_p_frames(mp4_path: Path) -> float:
    """Probes a Low-Delay P MP4 file to compute accurate TC_gt (P-frame bit average)."""
    if not mp4_path.exists():
        print(f"Warning: File {mp4_path} not found.")
        return 0.0
        
    cmd_probe_ldp = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pict_type,pkt_size", "-of", "csv=p=0", str(mp4_path)
    ]
    res = run_cmd(cmd_probe_ldp)
    
    p_sizes = []
    for line in res.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            p_type = None
            size = None
            for part in parts:
                if part.upper() in ["I", "P", "B"]:
                    p_type = part.upper()
                elif part.isdigit():
                    size = int(part)
            if p_type == "P" and size is not None:
                p_sizes.append(size)
                
    return float(np.mean(p_sizes)) * 8.0 if p_sizes else 0.0

# ==============================================================================
# STAGE 1: GROUND-TRUTH GENERATION (FFMPEG & FFPROBE)
# ==============================================================================
def generate_ground_truth(seq: dict, qp: int, force_reencode: bool = False) -> tuple[float, float, float]:
    """
    Encodes video in All-Intra (AI) and Low-Delay P (LDP) modes via x265,
    or re-uses existing encoded bitstreams if force_reencode is False.
    Returns: (SC_gt_bits, TC_gt_bits, T_enc_seconds)
    """
    seq_name = Path(seq["path"]).stem
    pix_fmt = "yuv420p" if seq["pix_fmt"] == "yuv420" else "yuv444p"
    
    # 1. All-Intra (AI) Encoding for SC_gt
    ai_mp4 = TEMP_DIR / f"{seq_name}_ai_qp{qp}.mp4"
    if force_reencode or not ai_mp4.exists():
        cmd_ai = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", pix_fmt,
            "-video_size", seq["res"], "-framerate", str(seq["fps"]),
            "-i", seq["path"], "-c:v", "libx265",
            "-x265-params", f"keyint=1:no-open-gop=1:qp={qp}",
            str(ai_mp4)
        ]
        run_cmd(cmd_ai)
    else:
        print(f"    [Re-using AI] {ai_mp4.name}")

    sc_gt = parse_ai_frame_sizes(ai_mp4)
    
    # 2. Low-Delay P (LDP) Encoding for TC_gt and Encoding Time
    ldp_mp4 = TEMP_DIR / f"{seq_name}_ldp_qp{qp}.mp4"
    t_enc = 0.0
    if force_reencode or not ldp_mp4.exists():
        cmd_ldp = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", pix_fmt,
            "-video_size", seq["res"], "-framerate", str(seq["fps"]),
            "-i", seq["path"], "-c:v", "libx265",
            "-x265-params", f"keyint=9999:bframes=0:no-scenecut=1:qp={qp}",
            "-benchmark", str(ldp_mp4)
        ]
        t0 = time.time()
        run_cmd(cmd_ldp)
        t_enc = time.time() - t0
    else:
        print(f"    [Re-using LDP] {ldp_mp4.name}")

    tc_gt = reparse_ldp_p_frames(ldp_mp4)
    
    return sc_gt, tc_gt, t_enc

# ==============================================================================
# STAGE 2: EVCA FEATURE EXTRACTION
# ==============================================================================
def run_evca_extraction(seq: dict) -> dict:
    """
    Executes EVCA CLI for 'baseline', 'fast', and 'full' profiles.
    Returns averaged metric scores across frames f >= 1.
    """
    seq_name = Path(seq["path"]).stem
    results = {}
    
    profiles = {
        "baseline": [sys.executable, EVCA_MAIN_PATH, "-i", seq["path"], "-r", seq["res"], "-p", seq["pix_fmt"], "--bit_depth", str(seq["bit_depth"])],
        "fast": [sys.executable, EVCA_MAIN_PATH, "-i", seq["path"], "-r", seq["res"], "-p", seq["pix_fmt"], "--bit_depth", str(seq["bit_depth"]), "-me", "-cc", "-cf", "--profile", "fast"],
        "full": [sys.executable, EVCA_MAIN_PATH, "-i", seq["path"], "-r", seq["res"], "-p", seq["pix_fmt"], "--bit_depth", str(seq["bit_depth"]), "-me", "-cc", "-cf", "--profile", "full"]
    }
    
    for prof_name, base_cmd in profiles.items():
        csv_out = TEMP_DIR / f"{seq_name}_evca_{prof_name}.csv"
        cmd = base_cmd + ["-c", str(csv_out)]
        run_cmd(cmd)
        
        df = pd.read_csv(csv_out)
        # Exclude frame 0 (reference frame initialization)
        df_valid = df.iloc[1:]
        
        for col in df_valid.columns:
            key = f"{prof_name}_{col}"
            results[key] = df_valid[col].mean()
            
    return results

# ==============================================================================
# STAGE 3: STATISTICAL CORRELATION & AGGREGATION (PATCHED)
# ==============================================================================
def compute_correlations(df_gt: pd.DataFrame, df_evca: pd.DataFrame) -> pd.DataFrame:
    """Computes Pearson (PCC) and Spearman (SRCC) metrics against SC_gt and TC_gt."""
    corr_records = []
    
    for qp in QPS:
        df_gt_qp = df_gt[df_gt["qp"] == qp]
        merged = pd.merge(df_gt_qp, df_evca, on="seq_name")
        
        # Temporal Metrics vs TC_gt
        tc_metrics = ["baseline_TC", "fast_TC_SAD", "fast_MVC", "full_TC_MC"]
        for metric in tc_metrics:
            if metric in merged.columns and not merged[metric].isnull().all():
                pcc, _ = pearsonr(merged[metric], merged["TC_gt"])
                srcc, _ = spearmanr(merged[metric], merged["TC_gt"])
                corr_records.append({
                    "Domain": "Temporal",
                    "QP": qp,
                    "Metric": metric,
                    "PCC": pcc,
                    "SRCC": srcc
                })
                
        # Spatial Metrics vs SC_gt
        sc_metrics = ["baseline_SC", "fast_SC_u", "fast_SC_v", "fast_Colorfulness"]
        for metric in sc_metrics:
            if metric in merged.columns and not merged[metric].isnull().all():
                pcc, _ = pearsonr(merged[metric], merged["SC_gt"])
                srcc, _ = spearmanr(merged[metric], merged["SC_gt"])
                corr_records.append({
                    "Domain": "Spatial",
                    "QP": qp,
                    "Metric": metric,
                    "PCC": pcc,
                    "SRCC": srcc
                })

    df_corr = pd.DataFrame(corr_records)
    df_corr.to_csv(CORR_CSV, index=False)
    return df_corr

# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="EVCA Full Validation Pipeline with Ground Truth Caching.")
    parser.add_argument(
        "--force-reencode",
        action="store_true",
        help="Force re-encoding and ignore cached ground truth MP4s."
    )
    args = parser.parse_args()

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Stage 1 & 2: Processing Sequences (force_reencode={args.force_reencode})...")
    gt_data = []
    evca_data = []
    
    for seq in TEST_SEQUENCES:
        seq_name = Path(seq["path"]).stem
        print(f"\n-> Processing: {seq_name}")
        
        # EVCA Feature Extraction
        print("  [EVCA] Extracting features...")
        evca_metrics = run_evca_extraction(seq)
        evca_metrics["seq_name"] = seq_name
        evca_data.append(evca_metrics)
        
        # Ground Truth Encodings across QPs
        for qp in QPS:
            print(f"  [Ground Truth] QP {qp}")
            sc_gt, tc_gt, t_enc = generate_ground_truth(seq, qp, force_reencode=args.force_reencode)
            gt_data.append({
                "seq_name": seq_name,
                "qp": qp,
                "SC_gt": sc_gt,
                "TC_gt": tc_gt,
                "T_enc": t_enc
            })
            
    df_gt = pd.DataFrame(gt_data)
    df_evca = pd.DataFrame(evca_data)
    
    # Repair any zero-value TC_gt entries from existing bitstreams if needed
    need_save = False
    for idx, row in df_gt.iterrows():
        if row["TC_gt"] == 0.0:
            seq_name = row["seq_name"]
            qp = int(row["qp"])
            ldp_file = TEMP_DIR / f"{seq_name}_ldp_qp{qp}.mp4"
            
            print(f"Repairing TC_gt for {seq_name} (QP {qp}) from {ldp_file.name}...")
            fixed_tc_gt = reparse_ldp_p_frames(ldp_file)
            print(f"  -> Extracted TC_gt: {fixed_tc_gt:.2f} bits/frame")
            df_gt.at[idx, "TC_gt"] = fixed_tc_gt
            need_save = True

    df_gt.to_csv(GT_CSV, index=False)
    df_evca.to_csv(EVCA_CSV, index=False)
    if need_save:
        print("Updated ground_truth_results.csv with repaired values.")
        
    print("\nStage 3: Calculating Correlation Matrices (PCC & SRCC)...")
    df_corr = compute_correlations(df_gt, df_evca)
    
    print("\n=== CORRELATION BENCHMARK SUMMARY ===")
    print(df_corr.to_string(index=False))

if __name__ == "__main__":
    main()
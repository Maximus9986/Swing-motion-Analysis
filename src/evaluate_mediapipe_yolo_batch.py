"""
evaluate_mediapipe_yolo_batch.py
---------------------------------
Runs MediaPipe + YOLO pipeline on all videos in VIDEO_DIR
and saves predictions to CSV for comparison with SwingNet.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_DIR   = r"C:\Users\User\FYP\Swing-motion-Analysis\Data\video_160"
SRC_DIR     = r"C:\Users\User\FYP\Swing-motion-Analysis\src"
YOLO_MODEL  = r"C:\Users\User\FYP\Swing-motion-Analysis\src\models\best.pt"
OUTPUT_CSV  = r"C:\Users\User\FYP\Swing-motion-Analysis\src\mediapipe_yolo_predictions.csv"
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, SRC_DIR)
from pose_tracking import extract_pose
from swing_analysis import analyze_swing
from club_tracking import extract_clubhead_y_from_yolo

all_videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])
print(f"Found {len(all_videos)} videos in {VIDEO_DIR}")
print(f"YOLO model: {'FOUND' if os.path.exists(YOLO_MODEL) else 'NOT FOUND'} -> {YOLO_MODEL}")

results = []
total = len(all_videos)

for i, filename in enumerate(all_videos):
    video_path = os.path.join(VIDEO_DIR, filename)
    vid_id     = int(filename.replace('.mp4', ''))

    print(f"\n[{i+1}/{total}] {filename}", flush=True)

    try:
        # ── Get FPS ──────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  {total_frames} frames @ {fps:.1f} fps", flush=True)

        # ── MediaPipe pose extraction ─────────────────────────────────────────
        print(f"  Running MediaPipe...", flush=True)
        pose_rows = extract_pose(video_path, output_path=None, write_overlay=False, use_3d=False)
        if not pose_rows:
            print(f"  SKIP: pose extraction returned nothing")
            continue

        pose_df = pd.DataFrame(pose_rows)
        if "frame" not in pose_df.columns:
            pose_df["frame"] = np.arange(len(pose_df))
        print(f"  Pose rows: {len(pose_df)}", flush=True)

        # ── YOLO club detection ───────────────────────────────────────────────
        club_df = None
        if os.path.exists(YOLO_MODEL):
            print(f"  Running YOLO...", flush=True)
            try:
                club_df = extract_clubhead_y_from_yolo(
                    video_path,
                    YOLO_MODEL,
                    conf=0.25,
                    clubhead_kpt_idx=0
                )
                valid_detections = int(club_df['clubhead_valid'].sum())
                print(f"  YOLO detections: {valid_detections}/{len(club_df)}", flush=True)
            except Exception as e:
                print(f"  YOLO failed: {e} — continuing with wrist-only", flush=True)
                club_df = None
        else:
            print(f"  YOLO model not found — using wrist-only impact", flush=True)

        # ── Merge club data if available ──────────────────────────────────────
        df = pose_df.copy()
        if club_df is not None:
            df = df.merge(
                club_df[["frame", "clubhead_y", "clubhead_valid", "clubhead_y_smooth"]],
                on="frame",
                how="left"
            )

        # ── Swing analysis ────────────────────────────────────────────────────
        print(f"  Running swing analysis...", flush=True)
        result_df = analyze_swing(df, fps=fps, debug=False)

        if result_df is None or result_df.empty:
            print(f"  SKIP: swing analysis returned empty")
            continue

        pred_top        = int(result_df["backswing_top_idx"].iloc[0])
        pred_impact     = int(result_df["impact_idx"].iloc[0])
        pred_impact_raw = int(result_df["impact_raw_idx"].iloc[0]) if "impact_raw_idx" in result_df.columns else pred_impact
        impact_method   = result_df["impact_method"].iloc[0] if "impact_method" in result_df.columns else "wrist"
        address_idx     = int(result_df["address_idx"].iloc[0])
        finish_idx      = int(result_df["finish_idx"].iloc[0])
        tempo_ratio     = result_df["tempo_ratio"].iloc[0] if "tempo_ratio" in result_df.columns else None
        overall_score   = int(result_df["overall_score"].iloc[0]) if "overall_score" in result_df.columns else None

        print(f"  address={address_idx}  top={pred_top}  impact={pred_impact}  finish={finish_idx}  method={impact_method}", flush=True)

        results.append({
            "id":                  vid_id,
            "filename":            filename,
            "total_frames":        total_frames,
            "fps":                 round(fps, 2),
            "mediapipe_address":   address_idx,
            "mediapipe_top":       pred_top,
            "mediapipe_impact":    pred_impact,
            "mediapipe_impact_raw": pred_impact_raw,
            "mediapipe_finish":    finish_idx,
            "impact_method":       impact_method,
            "tempo_ratio":         tempo_ratio,
            "overall_score":       overall_score,
        })

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}", flush=True)
        traceback.print_exc()
        continue

# ── Save CSV ──────────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n{'='*60}")
print(f"Done. {len(res_df)}/{total} videos processed.")
print(f"Saved to: {OUTPUT_CSV}")
print(f"\n{res_df[['id','mediapipe_top','mediapipe_impact','impact_method']].to_string()}")
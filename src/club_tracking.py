import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.signal import savgol_filter

# -----------------------------
# Shared helpers
# -----------------------------
def _safe_savgol(series, default_window=7, poly=2):
    """
    Safely apply Savitzky-Golay smoothing.
    Handles short clips + NaNs.
    """
    s = pd.Series(series).astype(float)
    s = s.interpolate(limit=5).ffill().bfill()
    n = len(s)
    if n < 5:
        return s.values
    w = min(default_window, n)
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return s.values
    poly = min(poly, w - 1)
    try:
        return savgol_filter(s.values, w, poly, mode="nearest")
    except Exception:
        return s.values


def extract_clubhead_y_from_yolo(video_path, model_path, conf=0.25, clubhead_kpt_idx=0):
    """
    Returns DataFrame with per-frame clubhead y (pixels):
    columns: frame, clubhead_y, clubhead_valid, clubhead_y_smooth
    - If model is YOLO-pose: uses keypoints[clubhead_kpt_idx]
    - Else: fallback uses bottom of bbox (y2)
    """
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, conf=conf, verbose=False)[0]

        y_val = np.nan
        valid = 0

        # Pose keypoints path
        if res.keypoints is not None and len(res.keypoints) > 0 and res.boxes is not None and len(res.boxes) > 0:
            best = int(np.argmax(res.boxes.conf.cpu().numpy()))
            kpts = res.keypoints.xy[best].cpu().numpy()  # (K,2)
            if clubhead_kpt_idx < len(kpts):
                y_val = float(kpts[clubhead_kpt_idx, 1])
                valid = 1

        # bbox fallback path
        elif res.boxes is not None and len(res.boxes) > 0:
            best = int(np.argmax(res.boxes.conf.cpu().numpy()))
            box = res.boxes.xyxy[best].cpu().numpy()
            y_val = float(box[3])  # y2 bottom
            valid = 1

        rows.append({"frame": frame_idx, "clubhead_y": y_val, "clubhead_valid": valid})
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"YOLO processed {frame_idx}/{total} frames...")

    cap.release()

    df = pd.DataFrame(rows)
    df["clubhead_y_smooth"] = _safe_savgol(df["clubhead_y"].values)
    return df


def refine_impact_with_club_y(impact_wrist, club_y, club_valid=None,
                              back_window=5, forward_window=12):
    """
    Refine a wrist-based impact frame using clubhead Y around it.
    Picks the LOWEST clubhead point (max y) within an asymmetric window.

    The window is biased forward because the wrist-based estimate almost
    always fires early (wrist leads the clubhead into impact).

    Parameters
    ----------
    impact_wrist : int
        Wrist-based impact frame estimate.
    club_y : np.array
        Per-frame clubhead Y values (length N, NaNs allowed).
    club_valid : np.array or None
        Per-frame 0/1 validity flags (length N).
    back_window : int
        How many frames to search before the wrist estimate (default 5).
    forward_window : int
        How many frames to search after the wrist estimate (default 12).
    """
    n = len(club_y)
    c = int(np.clip(impact_wrist, 0, n - 1))

    lo = max(0, c - back_window)
    hi = min(n, c + forward_window + 1)

    seg = club_y[lo:hi]

    if club_valid is None:
        seg_valid = ~np.isnan(seg)
        idxs = np.where(seg_valid)[0]
    else:
        seg_valid = np.asarray(club_valid[lo:hi]).astype(int)
        idxs = np.where(seg_valid == 1)[0]

    if len(idxs) == 0:
        return impact_wrist, "wrist_only"

    best_off = int(idxs[np.argmax(seg[idxs])])  # max y => lowest on screen
    return int(lo + best_off), "yolo_refine"
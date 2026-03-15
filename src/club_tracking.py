import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.signal import savgol_filter

def _safe_savgol(series, default_window=7, poly=2):
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
    Returns DataFrame with:
    frame, clubhead_y, clubhead_valid, clubhead_y_smooth
    """
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

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
            kpts = res.keypoints.xy[best].cpu().numpy()
            if clubhead_kpt_idx < len(kpts):
                y_val = float(kpts[clubhead_kpt_idx, 1])
                valid = 1

        # bbox fallback
        elif res.boxes is not None and len(res.boxes) > 0:
            best = int(np.argmax(res.boxes.conf.cpu().numpy()))
            box = res.boxes.xyxy[best].cpu().numpy()
            y_val = float(box[3])
            valid = 1

        rows.append({"frame": frame_idx, "clubhead_y": y_val, "clubhead_valid": valid})
        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    df["clubhead_y_smooth"] = _safe_savgol(df["clubhead_y"].values)
    return df
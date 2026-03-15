import numpy as np
import pandas as pd
import cv2
from scipy.signal import find_peaks, savgol_filter
from ultralytics import YOLO

# -----------------------------
# Helpers
# -----------------------------
def _safe_savgol(series, default_window=7, poly=2):
    """
    Safely apply Savitzky-Golay smoothing.
    Handles short clips + NaNs.
    """
    s = pd.Series(series).astype(float).ffill().bfill()
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
        return savgol_filter(s.values, window_length=w, polyorder=poly, mode="nearest")
    except Exception:
        return s.values


def _angle(p1, p2, p3):
    """Angle at p2 formed by p1-p2-p3 (in degrees)."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 180.0
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1, 1)
    return float(np.degrees(np.arccos(cosang)))


def _rolling_std(x, win=20):
    return pd.Series(x).rolling(win, center=True).std().bfill().ffill().values


def _dist2d(ax, ay, bx, by):
    return float(np.hypot(ax - bx, ay - by))


def _clamp01(x):
    return float(max(0.0, min(1.0, x)))
def extract_clubhead_y_from_yolo(video_path, model_path, conf=0.25, clubhead_kpt_idx=0):
    """
    Returns a DataFrame with per-frame clubhead y (pixels):
    columns: frame, clubhead_y, clubhead_valid
    - If model is YOLO-pose: uses keypoints[clubhead_kpt_idx]
    - Else: fallback uses bottom of bbox (y2)
    """
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    f = 0

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

        rows.append({"frame": f, "clubhead_y": y_val, "clubhead_valid": valid})
        f += 1

        if f % 50 == 0:
            print(f"YOLO processed {f}/{total} frames...")

    cap.release()
    return pd.DataFrame(rows)

def refine_impact_with_club_y(impact_wrist, club_y, club_valid=None, window=5):
    """
    Refine a wrist-based impact frame using clubhead Y around it.
    Picks the LOWEST clubhead point (max y) within +/- window frames.

    club_y: np.array length N (NaNs allowed)
    club_valid: np.array length N of 0/1 (optional). If None, valid = ~isnan(club_y).
    """
    n = len(club_y)
    c = int(np.clip(impact_wrist, 0, n - 1))

    lo = max(0, c - window)
    hi = min(n, c + window + 1)

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
    return int(lo + best_off), "yolo_refine_pm5"

# -----------------------------
# Early Extension (Side-view proxy)
# -----------------------------
def compute_early_extension(df, address_idx, impact_idx, debug=False):
    """
    Early extension proxy (side view):
    - hip drift (x) from address->impact
    - hip rise (y) from address->impact
    Normalized by torso length at address (shoulder->hip).
    Returns (score 0..100, label, debug_dict)
    """

    required = all(c in df.columns for c in [
        "hip_x_smooth", "hip_y_smooth",
        "shoulder_x_smooth", "shoulder_y_smooth"
    ])
    if not required:
        return 0, "N/A (missing hip/shoulder)", {}

    n = len(df)
    a = int(max(0, min(n - 1, address_idx)))
    i = int(max(0, min(n - 1, impact_idx)))

    hx = df["hip_x_smooth"].values.astype(float)
    hy = df["hip_y_smooth"].values.astype(float)
    sx = df["shoulder_x_smooth"].values.astype(float)
    sy = df["shoulder_y_smooth"].values.astype(float)

    torso = _dist2d(sx[a], sy[a], hx[a], hy[a])
    torso = max(torso, 1e-6)

    dx_hip = float(hx[i] - hx[a])
    dy_hip = float(hy[i] - hy[a])

    dxn = abs(dx_hip) / torso
    dyn = abs(dy_hip) / torso

    # Tunable thresholds (normalized)
    DX_OK, DX_BAD = 0.04, 0.10
    DY_OK, DY_BAD = 0.03, 0.08

    risk_x = _clamp01((dxn - DX_OK) / (DX_BAD - DX_OK))
    risk_y = _clamp01((dyn - DY_OK) / (DY_BAD - DY_OK))

    risk = 0.60 * risk_x + 0.40 * risk_y
    score = int(round(100 * risk))

    if score >= 70:
        label = "High early extension risk"
    elif score >= 40:
        label = "Moderate early extension risk"
    else:
        label = "Low early extension risk"

    dbg = {
        "dx_hip": dx_hip, "dy_hip": dy_hip,
        "torso_ref": torso,
        "dx_norm": dxn, "dy_norm": dyn,
        "risk_x": risk_x, "risk_y": risk_y,
        "score": score
    }

    if debug:
        print("EARLY EXT DEBUG:", dbg)

    return score, label, dbg


# -----------------------------
# Main analysis
# -----------------------------
def analyze_swing(df, fps=None, debug=False):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # -----------------------------
    # Safe defaults (never KeyError)
    # -----------------------------
    df["address_idx"] = 0
    df["backswing_start_idx"] = 0
    df["backswing_top_idx"] = 0
    df["impact_raw_idx"] = 0
    df["impact_idx"] = 0
    df["finish_idx"] = max(len(df) - 1, 0)
    df["impact_method"] = "unset"

    # -----------------------------
    # 0) 3D check
    # -----------------------------
    has_3d = ("wrist_z" in df.columns) or ("wrist_z_smooth" in df.columns)
    df["has_3d_data"] = bool(has_3d)

    # -----------------------------
    # 1) Smooth joints
    # -----------------------------
    for joint in ["wrist", "elbow", "shoulder", "hip"]:
        for axis in ["x", "y", "z"]:
            col = f"{joint}_{axis}"
            if col in df.columns:
                df[f"{col}_smooth"] = _safe_savgol(df[col].values)

    if "wrist_y_smooth" not in df.columns:
        df["error"] = "Missing wrist_y"
        return df

    y = df["wrist_y_smooth"].values.astype(float)
    n = len(y)

    # -----------------------------
    # 2) Address + backswing start
    # -----------------------------
    def detect_address_and_backswing_start(yvals):
        n_ = len(yvals)
        if n_ < 30:
            return 0, 0

        total_range = np.nanpercentile(yvals, 95) - np.nanpercentile(yvals, 5)
        total_range = max(total_range, 1e-6)

        MIN_DROP = 0.2
        local_drop = 0.02 * total_range

        address = 0
        for i in range(20, min(200, n_ - 40)):
            if yvals[i] < yvals[address] - local_drop:
                if yvals[address] - np.nanmin(yvals[i:i + 40]) >= MIN_DROP:
                    return address, max(0, i - 5)
        return 0, 0

    address_idx, backswing_start = detect_address_and_backswing_start(y)
    df["address_idx"] = int(address_idx)
    df["backswing_start_idx"] = int(backswing_start)

    # -----------------------------
    # 3) Top of backswing 
    # -----------------------------
    search_window = min(n - backswing_start - 1, 500)
    segment = y[backswing_start : backswing_start + search_window]

    seg_range = float(np.max(segment) - np.min(segment))
    prom = 0.15 * max(seg_range, 1e-6)

    troughs, properties = find_peaks(
        -segment,
        prominence=prom,
        distance=8
    )

    # Minimum Y drop required from backswing_start to be considered valid backswing
    MIN_DROP_FROM_START = 0.35
    LOOKAHEAD = 25

    # Y value at backswing start (address level)
    y_at_start = y[backswing_start]

    backswing_top = None

    if len(troughs) > 0:
        for t in troughs:
            trough_idx = backswing_start + t
            y_at_trough = y[trough_idx]
            
            # Check 1: Must have dropped at least MIN_DROP_FROM_START from start
            drop_from_start = y_at_start - y_at_trough
            
            if drop_from_start < MIN_DROP_FROM_START:
                # This trough hasn't dropped enough - likely just jitter
                continue
            
            # Check 2: Must have significant movement after this point
            j = min(t + LOOKAHEAD, len(segment) - 1)
            move_after = float(np.max(segment[t:j+1]) - np.min(segment[t:j+1]))

            if move_after >= 0.1:  # Some movement after the top
                backswing_top = int(trough_idx)
                break

    # Fallback 1: Find deepest trough that meets minimum drop requirement
    if backswing_top is None and len(troughs) > 0:
        valid_troughs = []
        for t in troughs:
            trough_idx = backswing_start + t
            y_at_trough = y[trough_idx]
            drop_from_start = y_at_start - y_at_trough
            
            if drop_from_start >= MIN_DROP_FROM_START:
                valid_troughs.append(t)
        
        if valid_troughs:
            # Pick the deepest valid trough
            best = int(valid_troughs[np.argmin([segment[t] for t in valid_troughs])])
            backswing_top = int(backswing_start + best)

    # Fallback 2: Find absolute minimum in segment, but only if it meets drop requirement
    if backswing_top is None:
        min_idx = int(np.argmin(segment))
        y_at_min = segment[min_idx]
        drop_from_start = y_at_start - y_at_min
        
        if drop_from_start >= MIN_DROP_FROM_START:
            backswing_top = int(backswing_start + min_idx)
        else:
            # No valid backswing detected - use start as fallback
            backswing_top = int(backswing_start)
            if debug:
                print(f"WARNING: No backswing detected with drop >= {MIN_DROP_FROM_START}")

    print(f"DEBUG: Backswing start = {backswing_start}")
    print(f"DEBUG: Y at start = {y_at_start:.4f}")
    print(f"DEBUG: Backswing top = {backswing_top}")
    print(f"DEBUG: Y at top = {y[backswing_top]:.4f}")
    print(f"DEBUG: Drop from start = {y_at_start - y[backswing_top]:.4f}")
    df["backswing_top_idx"] = int(backswing_top)


    # -----------------------------
    # 4) Impact (WRIST → YOLO refine)
    # -----------------------------
    post = y[backswing_top: min(backswing_top + 120, n)]
    peaks, _ = find_peaks(post, prominence=0.12 * np.ptp(post))

    if len(peaks):
        impact_wrist = backswing_top + peaks[0]
    else:
        impact_wrist = backswing_top + int(np.nanargmax(post))

    impact = int(impact_wrist)
    impact_method = "wrist"

    if "clubhead_y_smooth" in df.columns:
        club_y = df["clubhead_y_smooth"].values.astype(float)
        club_valid = df["clubhead_valid"].values.astype(int) if "clubhead_valid" in df.columns else None
        impact, impact_method = refine_impact_with_club_y(
            impact_wrist, club_y, club_valid, window=5
        )

    df["impact_raw_idx"] = int(impact_wrist)
    df["impact_idx"] = int(impact)
    df["impact_method"] = impact_method
    # -----------------------------
    # 4.5) Finish detection (ROM based)
    # -----------------------------
    finish = n - 1

    # Start searching a few frames after impact
    START_OFFSET = 6
    WINDOW = 10           # frames to check stability
    STABLE_RANGE = 0.05   # wrist Y must stay within this range

    start = min(impact + START_OFFSET, n - WINDOW - 1)

    for i in range(start, n - WINDOW):
        seg = y[i:i + WINDOW]

        # range of wrist Y in this window
        r = float(np.nanmax(seg) - np.nanmin(seg))

        if r < STABLE_RANGE:
            finish = i
            break

    df["finish_idx"] = int(finish)
    df["follow_through_frames"] = int(max(finish - impact, 0))

    # -----------------------------
    # 5) Elbow angle
    # -----------------------------
    elbow_angle = None
    if all(c in df.columns for c in [
        "shoulder_x_smooth", "shoulder_y_smooth",
        "elbow_x_smooth", "elbow_y_smooth",
        "wrist_x_smooth", "wrist_y_smooth"
    ]):
        shoulder = [df.loc[impact, "shoulder_x_smooth"], df.loc[impact, "shoulder_y_smooth"], 0]
        elbow    = [df.loc[impact, "elbow_x_smooth"],    df.loc[impact, "elbow_y_smooth"],    0]
        wrist    = [df.loc[impact, "wrist_x_smooth"],    df.loc[impact, "wrist_y_smooth"],    0]
        elbow_angle = _angle(shoulder, elbow, wrist)

    df["elbow_angle_impact"] = elbow_angle if elbow_angle else np.nan
    df["arm_extension_label"] = (
        "Excellent" if elbow_angle and elbow_angle > 155 else
        "Good" if elbow_angle and elbow_angle > 140 else
        "Moderate" if elbow_angle and elbow_angle > 125 else
        "Needs improvement"
    )

    # -----------------------------
    # 6) Wrist speed + timing
    # -----------------------------
    fps_used = fps if fps else 30.0
    dt = 1.0 / fps_used

    wx = df["wrist_x_smooth"].values
    wy = df["wrist_y_smooth"].values

    vel = np.sqrt(np.diff(wx)**2 + np.diff(wy)**2) / dt
    seg = vel[backswing_top: min(impact + 8, len(vel))]

    max_v_frame = backswing_top + int(np.nanargmax(seg)) if len(seg) else impact
    frames_from_impact = max_v_frame - impact

    df["max_speed_frame"] = int(max_v_frame)
    df["max_speed_frames_from_impact"] = int(frames_from_impact)

    df["speed_timing_score"] = (
        100 if abs(frames_from_impact) <= 2 else
        85 if abs(frames_from_impact) <= 4 else
        70 if abs(frames_from_impact) <= 6 else
        50
    )

    # -----------------------------
    # 7) Tempo
    # -----------------------------
    backswing_frames = backswing_top - backswing_start
    downswing_frames = max(impact - backswing_top, 1)
    tempo_ratio = round(backswing_frames / downswing_frames, 2)

    df["backswing_frames"] = int(backswing_frames)
    df["downswing_frames"] = int(downswing_frames)
    df["tempo_ratio"] = tempo_ratio

    # -----------------------------
    # 8) Early extension
    # -----------------------------
    ee_score, ee_label, _ = compute_early_extension(df, address_idx, impact)
    df["early_extension_score"] = ee_score
    df["early_extension_label"] = ee_label

    # -----------------------------
    # 9) Overall score
    # -----------------------------
    score = 0

    score += 25 if 2.0 <= tempo_ratio <= 3.5 else 18 if 1.5 <= tempo_ratio <= 4.0 else 10
    score += 25 if elbow_angle and elbow_angle > 155 else 18 if elbow_angle and elbow_angle > 140 else 10
    score += int(df["speed_timing_score"].iloc[0] * 0.25)
    score += 25 if ee_score <= 30 else 18 if ee_score <= 60 else 10

    df["overall_score"] = int(score)
    df["overall_rating"] = (
        "Excellent" if score >= 85 else
        "Good" if score >= 70 else
        "Average" if score >= 55 else
        "Needs Work"
    )

    return df
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Fix 1: Import shared helpers from club_tracking (single source of truth)
from club_tracking import _safe_savgol, refine_impact_with_club_y


# -----------------------------
# Helpers (swing-analysis specific)
# -----------------------------
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
    # 2) Find backswing top FIRST
    # -----------------------------
    search_window = min(n - 1, 500)
    segment = y[0 : search_window]

    seg_range = float(np.max(segment) - np.min(segment))
    prom = 0.15 * max(seg_range, 1e-6)

    troughs, properties = find_peaks(
        -segment,
        prominence=prom,
        distance=8
    )

    MIN_DROP_ABS = 0.35
    LOOKAHEAD = 25

    backswing_top = None
    # Use the median of the first quarter of the search window as the baseline.
    baseline_end = max(10, search_window // 4)
    baseline_y = float(np.nanmedian(y[0:baseline_end]))

    if len(troughs) > 0:
        for t in troughs:
            y_at_trough = y[t]
            drop_from_baseline = baseline_y - y_at_trough

            if drop_from_baseline < MIN_DROP_ABS:
                continue

            j = min(t + LOOKAHEAD, len(segment) - 1)
            move_after = float(np.max(segment[t:j+1]) - np.min(segment[t:j+1]))

            if move_after >= 0.1:
                backswing_top = int(t)
                break

    # Fallback 1: deepest valid trough
    if backswing_top is None and len(troughs) > 0:
        valid_troughs = []
        for t in troughs:
            y_at_trough = y[t]
            drop_from_baseline = baseline_y - y_at_trough
            if drop_from_baseline >= MIN_DROP_ABS:
                valid_troughs.append(t)

        if valid_troughs:
            best = int(valid_troughs[np.argmin([segment[t] for t in valid_troughs])])
            backswing_top = int(best)

    # Fallback 2: absolute minimum if it meets drop requirement
    if backswing_top is None:
        min_idx = int(np.argmin(segment))
        y_at_min = segment[min_idx]
        drop_from_baseline = baseline_y - y_at_min

        if drop_from_baseline >= MIN_DROP_ABS:
            backswing_top = int(min_idx)
        else:
            backswing_top = 0
            if debug:
                print(f"WARNING: No backswing detected with drop >= {MIN_DROP_ABS}")

    df["backswing_top_idx"] = int(backswing_top)

    # -----------------------------
    # 3) Address + backswing start
    # -----------------------------
    STABLE_WINDOW = 8        # frames to check for flatness
    STABLE_THRESH = 0.04     # max wrist Y range to count as "quiet"
    DROP_RATE_THRESH = 0.008 # per-frame drop rate that signals the swing has started
    ADDRESS_OFFSET = 5       # address is this many frames before backswing start

    address_idx = 0
    backswing_start = max(backswing_top - 1, 0)

    for i in range(backswing_top - 1, STABLE_WINDOW, -1):
        dy = y[i] - y[i + 1] 

        if dy < DROP_RATE_THRESH:
            win = y[max(0, i - STABLE_WINDOW):i + 1]
            win_range = float(np.nanmax(win) - np.nanmin(win))

            if win_range < STABLE_THRESH:
                backswing_start = int(i + 1)
                address_idx = int(max(0, backswing_start - ADDRESS_OFFSET))
                break
    else:
        # If we walked all the way back without finding a quiet region,
        # use a simple heuristic: find the frame with max Y before the top
        pre_top = y[0:backswing_top]
        if len(pre_top) > 0:
            backswing_start = int(np.argmax(pre_top)) + 1
            address_idx = int(max(0, backswing_start - ADDRESS_OFFSET))

    df["address_idx"] = int(address_idx)
    df["backswing_start_idx"] = int(backswing_start)

    if debug:
        print(f"DEBUG: Backswing top = {backswing_top}")
        print(f"DEBUG: Y at top = {y[backswing_top]:.4f}")
        print(f"DEBUG: Address = {address_idx}")
        print(f"DEBUG: Y at address = {y[address_idx]:.4f}")
        print(f"DEBUG: Backswing start = {backswing_start}")
        print(f"DEBUG: Drop from address = {y[address_idx] - y[backswing_top]:.4f}")

    # -----------------------------
    # 4) Impact (WRIST -> YOLO refine)
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
            impact_wrist, club_y, club_valid,
            back_window=5, forward_window=12
        )

    df["impact_raw_idx"] = int(impact_wrist)
    df["impact_idx"] = int(impact)
    df["impact_method"] = impact_method

    # -----------------------------
    # 4.5) Finish detection (Fix 6: improved fallback)
    # -----------------------------
    finish = n - 1

    START_OFFSET = 6
    WINDOW = 10
    STABLE_RANGE = 0.05

    start = min(impact + START_OFFSET, n - WINDOW - 1)

    found_stable = False
    for i in range(start, n - WINDOW):
        seg = y[i:i + WINDOW]
        r = float(np.nanmax(seg) - np.nanmin(seg))
        if r < STABLE_RANGE:
            finish = i
            found_stable = True
            break

    # Fix 6: If no stable window found, try a relaxed threshold,
    # then fall back to the point of minimum wrist speed after impact.
    if not found_stable:
        RELAXED_RANGE = 0.10
        for i in range(start, n - WINDOW):
            seg = y[i:i + WINDOW]
            r = float(np.nanmax(seg) - np.nanmin(seg))
            if r < RELAXED_RANGE:
                finish = i
                found_stable = True
                break

    if not found_stable:
        # Use the frame with the lowest wrist speed after impact+offset
        if "wrist_x_smooth" in df.columns:
            wx = df["wrist_x_smooth"].values
            wy = df["wrist_y_smooth"].values
            speed = np.sqrt(np.diff(wx)**2 + np.diff(wy)**2)
            search_start = min(impact + START_OFFSET, len(speed) - 1)
            if search_start < len(speed):
                finish = search_start + int(np.argmin(speed[search_start:]))
        # else: keep finish = n - 1

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

    speed_timing_score = (
        100 if abs(frames_from_impact) <= 2 else
        85 if abs(frames_from_impact) <= 4 else
        70 if abs(frames_from_impact) <= 6 else
        50
    )
    df["speed_timing_score"] = speed_timing_score

    # Fix 5: Add the text label that app.py references
    df["speed_timing"] = (
        "Excellent - peak speed at impact" if abs(frames_from_impact) <= 2 else
        "Good - peak speed near impact" if abs(frames_from_impact) <= 4 else
        "Moderate - peak speed slightly early" if frames_from_impact < -4 else
        "Moderate - peak speed slightly late" if frames_from_impact > 4 else
        "Needs work - peak speed far from impact"
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
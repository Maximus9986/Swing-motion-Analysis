import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


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
    """
    Swing analysis pipeline (side-view friendly).
    Produces these columns safely (always exists):
      - address_idx
      - backswing_start_idx
      - backswing_top_idx
      - impact_raw_idx
      - impact_idx
      - finish_idx
      - backswing_frames, downswing_frames, tempo_ratio, follow_through_frames
      - elbow_angle_impact, arm_extension_label
      - max_wrist_speed, max_speed_frame, speed_timing, speed_timing_score
      - early_extension_score, early_extension_label
      - overall_score, overall_rating
      - has_3d_data
    """

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Always create phase columns so Streamlit never KeyErrors
    df["address_idx"] = 0
    df["backswing_start_idx"] = 0
    df["backswing_top_idx"] = 0
    df["impact_raw_idx"] = 0
    df["impact_idx"] = 0
    df["finish_idx"] = max(len(df) - 1, 0)

    # -----------------------------
    # 0) Determine if 3D
    # -----------------------------
    has_3d = ("wrist_z" in df.columns) or ("wrist_z_smooth" in df.columns)
    df["has_3d_data"] = bool(has_3d)

    # -----------------------------
    # 1) Smooth coordinates
    # -----------------------------
    for joint in ["wrist", "elbow", "shoulder", "hip"]:
        for axis in ["x", "y", "z"]:
            col = f"{joint}_{axis}"
            if col in df.columns:
                df[f"{col}_smooth"] = _safe_savgol(df[col].values, default_window=7, poly=2)

    if "wrist_y_smooth" not in df.columns:
        df["error"] = "Missing wrist_y"
        return df

    y = df["wrist_y_smooth"].values.astype(float)
    n = len(y)

    # -----------------------------
    # 2) Address + backswing start detection
    # -----------------------------
    def detect_address_and_backswing_start(yvals):
        yvals = np.asarray(yvals, dtype=float)
        n_ = len(yvals)
        if n_ < 30:
            address_mean = float(np.nanmean(yvals[: min(10, n_)]))
            return 0, 0, min(10, n_ - 1), address_mean

        total_range = np.nanpercentile(yvals, 95) - np.nanpercentile(yvals, 5)
        total_range = max(float(total_range), 1e-6)

        # >>> Your rule: if swing never drops enough, don't treat it as backswing <<<
        # Using absolute minimum drop threshold (0.2) as you requested
        MIN_TOTAL_DROP = 0.2
        significant_drop = max(0.30 * total_range, MIN_TOTAL_DROP)

        win = 20
        rs = _rolling_std(yvals, win=win)

        max_search = min(n_ // 2, 250)
        stability_threshold = 0.06 * total_range
        min_len = 12

        stable_segments = []
        start = None
        length = 0
        for i in range(win, max_search):
            if rs[i] < stability_threshold:
                if start is None:
                    start = i
                length += 1
            else:
                if start is not None and length >= min_len:
                    stable_segments.append((start, i - 1, length))
                start = None
                length = 0

        if start is not None and length >= min_len:
            stable_segments.append((start, max_search - 1, length))

        if stable_segments:
            stable_segments.sort(key=lambda t: t[2], reverse=True)
            s0, s1, _ = stable_segments[0]
        else:
            s0, s1 = 0, min(20, max_search - 1)

        address_mean = float(np.nanmean(yvals[s0:s1 + 1]))

        local_drop = 0.02 * total_range
        confirm_window = 40

        for i in range(s1, min(s1 + 80, n_ - confirm_window - 1)):
            if yvals[i] < address_mean - local_drop:
                future_min = float(np.nanmin(yvals[i:i + confirm_window]))
                # require big enough drop (>= significant_drop, which is >= 0.2)
                if (address_mean - future_min) >= significant_drop:
                    backswing_start = max(i - 5, s1)
                    return s0, backswing_start, s1, address_mean

        # fallback: negative slope streak
        dy = np.diff(yvals)
        for i in range(20, min(250, len(dy) - 6)):
            if np.all(dy[i:i + 6] < -0.01 * total_range):
                backswing_start = i
                return s0, backswing_start, s1, address_mean

        return s0, 0, s1, address_mean

    address_idx, backswing_start, address_end, address_y = detect_address_and_backswing_start(y)

    df["address_idx"] = int(address_idx)
    df["backswing_start_idx"] = int(backswing_start)

    if debug:
        print("DEBUG address_idx:", address_idx, "backswing_start:", backswing_start, "address_y:", address_y)

    # -----------------------------
    # 3) Top of backswing detection (your tuned + drop threshold)
    # -----------------------------
    if backswing_start >= n - 5:
        df["error"] = "Backswing start too late"
        return df

    search_window = min(140, n - backswing_start - 1)  # allow slower swings
    segment = y[backswing_start: backswing_start + search_window]

    seg_range = float(np.nanmax(segment) - np.nanmin(segment))
    prom = 0.15 * max(seg_range, 1e-6)

    troughs, _ = find_peaks(-segment, prominence=prom, distance=8)

    MIN_DROP_FROM_START = 0.2  # your rule
    LOOKAHEAD = 25

    y_at_start = float(y[backswing_start])
    backswing_top = None

    if len(troughs) > 0:
        for t in troughs:
            trough_idx = int(backswing_start + t)
            y_at_trough = float(y[trough_idx])
            drop_from_start = y_at_start - y_at_trough

            if drop_from_start < MIN_DROP_FROM_START:
                continue

            j = min(t + LOOKAHEAD, len(segment) - 1)
            move_after = float(np.nanmax(segment[t:j+1]) - np.nanmin(segment[t:j+1]))
            if move_after >= 0.1:
                backswing_top = trough_idx
                break

    # fallback: deepest valid trough
    if backswing_top is None and len(troughs) > 0:
        valid = []
        for t in troughs:
            trough_idx = int(backswing_start + t)
            drop_from_start = y_at_start - float(y[trough_idx])
            if drop_from_start >= MIN_DROP_FROM_START:
                valid.append(t)

        if valid:
            best = int(valid[np.argmin([segment[t] for t in valid])])
            backswing_top = int(backswing_start + best)

    # fallback: absolute min IF meets drop requirement else backswing_start
    if backswing_top is None:
        min_idx = int(np.nanargmin(segment))
        y_at_min = float(segment[min_idx])
        drop_from_start = y_at_start - y_at_min
        if drop_from_start >= MIN_DROP_FROM_START:
            backswing_top = int(backswing_start + min_idx)
        else:
            backswing_top = int(backswing_start)
            if debug:
                print(f"WARNING: No valid backswing top (drop<{MIN_DROP_FROM_START})")

    df["backswing_top_idx"] = int(backswing_top)

    if debug:
        print("DEBUG backswing_top:", backswing_top, "y_top:", float(y[backswing_top]), "drop:", y_at_start - float(y[backswing_top]))

    # -----------------------------
    # 4) Impact detection (closest return to address_y after top)
    # -----------------------------
    search_start = int(backswing_top)
    search_end = int(min(backswing_top + 120, n))  # allow enough frames

    post = y[search_start:search_end]

    # Find peaks in post-top segment (peaks in wrist_y)
    seg_range = float(np.nanmax(post) - np.nanmin(post))
    prom = 0.12 * max(seg_range, 1e-6)  # tune 0.10~0.20 if needed

    peaks, props = find_peaks(
        post,
        prominence=prom,
        distance=8
    )

    if len(peaks) > 0:
        # "next peak after top"
        impact_raw = int(search_start + peaks[0])
    else:
        # fallback: choose biggest rise point (max y) in window
        impact_raw = int(search_start + int(np.nanargmax(post)))

    impact = int(min(max(impact_raw, 0), n - 1))

    df["impact_raw_idx"] = int(impact_raw)
    df["impact_idx"] = int(impact)
    # -----------------------------
    # 5) Arm extension at impact (elbow angle)
    # -----------------------------
    needed = all(c in df.columns for c in [
        "shoulder_x_smooth", "shoulder_y_smooth",
        "elbow_x_smooth", "elbow_y_smooth",
        "wrist_x_smooth", "wrist_y_smooth"
    ])

    elbow_angle = None
    if needed:
        if has_3d and all(c in df.columns for c in ["shoulder_z_smooth", "elbow_z_smooth", "wrist_z_smooth"]):
            shoulder = [df.loc[impact, "shoulder_x_smooth"], df.loc[impact, "shoulder_y_smooth"], df.loc[impact, "shoulder_z_smooth"]]
            elbow    = [df.loc[impact, "elbow_x_smooth"],    df.loc[impact, "elbow_y_smooth"],    df.loc[impact, "elbow_z_smooth"]]
            wrist    = [df.loc[impact, "wrist_x_smooth"],    df.loc[impact, "wrist_y_smooth"],    df.loc[impact, "wrist_z_smooth"]]
        else:
            shoulder = [df.loc[impact, "shoulder_x_smooth"], df.loc[impact, "shoulder_y_smooth"], 0.0]
            elbow    = [df.loc[impact, "elbow_x_smooth"],    df.loc[impact, "elbow_y_smooth"],    0.0]
            wrist    = [df.loc[impact, "wrist_x_smooth"],    df.loc[impact, "wrist_y_smooth"],    0.0]

        elbow_angle = _angle(shoulder, elbow, wrist)

        if elbow_angle > 155:
            arm_label = "Excellent (fully extended)"
        elif elbow_angle > 140:
            arm_label = "Good (slight bend is normal)"
        elif elbow_angle > 125:
            arm_label = "Moderate"
        else:
            arm_label = "Needs improvement (chicken wing)"

        df["elbow_angle_impact"] = float(elbow_angle)
        df["arm_extension_label"] = arm_label
    else:
        df["elbow_angle_impact"] = np.nan
        df["arm_extension_label"] = "N/A (missing joint data)"

    # -----------------------------
    # 6) Wrist speed + timing
    # -----------------------------
    fps_used = float(fps) if fps is not None else float(df.get("fps", pd.Series([30])).iloc[0] if "fps" in df.columns else 30.0)
    if fps_used <= 0:
        fps_used = 30.0
    dt = 1.0 / fps_used

    wrist_x = df["wrist_x_smooth"].values.astype(float) if "wrist_x_smooth" in df.columns else None
    wrist_y = df["wrist_y_smooth"].values.astype(float)

    if wrist_x is None:
        df["max_wrist_speed"] = 0.0
        df["max_speed_frame"] = int(impact)
        df["speed_timing"] = "N/A"
        df["speed_timing_score"] = 0
        df["max_speed_frames_from_impact"] = 0
    else:
        dx = np.diff(wrist_x) / dt
        dy = np.diff(wrist_y) / dt

        if has_3d and "wrist_z_smooth" in df.columns:
            wz = df["wrist_z_smooth"].values.astype(float)
            dz = np.diff(wz) / dt
            velocity = np.sqrt(dx * dx + dy * dy + dz * dz)
        else:
            velocity = np.sqrt(dx * dx + dy * dy)

        downswing_start_idx = int(backswing_top)
        downswing_end_idx = int(min(impact + 8, len(velocity)))

        seg = velocity[downswing_start_idx:downswing_end_idx]
        if len(seg) > 0:
            max_v = float(np.nanmax(seg))
            max_v_frame = int(downswing_start_idx + int(np.nanargmax(seg)))
        else:
            max_v = 0.0
            max_v_frame = int(impact)

        df["max_wrist_speed"] = max_v
        df["max_speed_frame"] = max_v_frame

        frames_from_impact = int(max_v_frame - impact)
        abs_diff = abs(frames_from_impact)

        if abs_diff <= 2:
            speed_timing = "Excellent (max speed at impact)"
            speed_score = 100
        elif abs_diff <= 4:
            speed_timing = "Good (max speed near impact)"
            speed_score = 85
        elif abs_diff <= 6:
            speed_timing = "Moderate"
            speed_score = 70
        else:
            speed_timing = "Early release" if frames_from_impact < 0 else "Late release"
            speed_score = 50

        df["speed_timing"] = speed_timing
        df["speed_timing_score"] = int(speed_score)
        df["max_speed_frames_from_impact"] = int(frames_from_impact)

    # -----------------------------
    # 7) Tempo
    # -----------------------------
    backswing_frames = int(max(backswing_top - backswing_start, 0))
    downswing_frames = int(max(impact - backswing_top, 1))
    tempo_ratio = round(backswing_frames / downswing_frames, 2) if downswing_frames > 0 else 0.0

    df["backswing_frames"] = backswing_frames
    df["downswing_frames"] = downswing_frames
    df["tempo_ratio"] = tempo_ratio

    # -----------------------------
    # 8) Finish detection (speed stabilisation)
    # -----------------------------
    finish = n - 1
    try:
        if wrist_x is not None:
            vx = np.diff(wrist_x) / dt
            vy = np.diff(wrist_y) / dt
            speed = np.sqrt(vx * vx + vy * vy)
            post = speed[min(impact, len(speed) - 1):]

            if len(post) >= 20:
                peak = float(np.nanmax(speed)) if len(speed) else 1.0
                thresh = 0.12 * max(peak, 1e-6)
                stable_len = 12
                count = 0
                for i_ in range(len(post)):
                    if post[i_] < thresh:
                        count += 1
                        if count >= stable_len:
                            finish = min(impact + i_, n - 1)
                            break
                    else:
                        count = 0
    except Exception:
        finish = n - 1

    df["finish_idx"] = int(finish)
    df["follow_through_frames"] = int(max(finish - impact, 0))

    # -----------------------------
    # 9) Early Extension (replaces hand path)
    # -----------------------------
    ee_score, ee_label, ee_dbg = compute_early_extension(df, int(df["address_idx"].iloc[0]), int(df["impact_idx"].iloc[0]), debug=debug)
    df["early_extension_score"] = int(ee_score)
    df["early_extension_label"] = ee_label
    df["hip_dx_norm"] = float(ee_dbg.get("dx_norm", np.nan))
    df["hip_dy_norm"] = float(ee_dbg.get("dy_norm", np.nan))

    # -----------------------------
    # 10) Overall score (tempo + extension + speed timing + early extension)
    # -----------------------------
    score = 0
    max_score = 100  # fixed

    # Tempo (25)
    if 2.0 <= tempo_ratio <= 3.5:
        score += 25
    elif 1.5 <= tempo_ratio <= 4.0:
        score += 18
    else:
        score += 10

    # Arm extension (25)
    if elbow_angle is not None:
        if elbow_angle > 155:
            score += 25
        elif elbow_angle > 140:
            score += 22
        elif elbow_angle > 125:
            score += 15
        else:
            score += 8
    else:
        score += 12

    # Speed timing (25)
    speed_timing_score = int(df["speed_timing_score"].iloc[0]) if "speed_timing_score" in df.columns else 0
    score += int(speed_timing_score * 0.25)

    # Early extension (25) -> lower is better
    if ee_score <= 30:
        score += 25
    elif ee_score <= 60:
        score += 18
    else:
        score += 10

    overall = int(round(score / max_score * 100))

    if overall >= 85:
        rating = "Excellent"
    elif overall >= 70:
        rating = "Good"
    elif overall >= 55:
        rating = "Average"
    else:
        rating = "Needs Work"

    df["overall_score"] = int(overall)
    df["overall_rating"] = rating

    return df

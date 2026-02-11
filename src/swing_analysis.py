import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


def _safe_savgol(series, default_window=7, poly=2):

    s = pd.Series(series).astype(float).ffill().bfill()

    n = len(s)
    if n < 5:
        return s.values

    # Window must be odd and <= n
    w = min(default_window, n)
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return s.values

    # polyorder must be < window
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


# -----------------------------
# Main analysis
# -----------------------------
def analyze_swing(df, club_type="iron", fps=None, debug=False):
    """
    Robust swing analysis pipeline.
    Works with MediaPipe 2D (x,y) and optional 3D (x,y,z).
    Adds:
      - address_idx
      - backswing_start_idx
      - backswing_top_idx
      - impact_raw_idx
      - impact_idx
      - finish_idx
      - tempo_ratio, backswing_frames, downswing_frames, follow_through_frames
      - hand_path_steepness, hand_path_label, has_3d_data
      - elbow_angle_impact, arm_extension_label
      - max_wrist_speed, max_speed_frame, speed_timing, speed_timing_score
      - overall_score, overall_rating
    """

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    club_type = (club_type or "iron").lower().strip()
    df["club_type"] = club_type
    df["backswing_top_idx"] = 0
    df["impact_raw_idx"] = 0
    df["impact_idx"] = 0
    df["finish_idx"] = len(df) - 1
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

    # Need wrist_y_smooth
    if "wrist_y_smooth" not in df.columns:
        df["error"] = "Missing wrist_y column"
        return df

    y = df["wrist_y_smooth"].values.astype(float)
    n = len(y)


    # -----------------------------
    # 2) Robust address + backswing start detection
    # -----------------------------
    def detect_address_and_backswing_start(yvals):
        yvals = np.asarray(yvals, dtype=float)
        n = len(yvals)
        if n < 30:
            return 0, 0, min(10, n - 1), float(np.nanmean(yvals[: min(10, n)]))

        # Estimate movement range
        total_range = np.nanpercentile(yvals, 95) - np.nanpercentile(yvals, 5)
        significant_drop = 0.30 * total_range  # must drop a lot after start

        # Find stable "address" segment early in video
        win = 20
        rs = _rolling_std(yvals, win=win)

        max_search = min(n // 2, 250)
        stability_threshold = 0.06 * max(total_range, 1e-6)  # scale with range
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

        # Backswing start: first consistent move away from address that later leads to large drop
        local_drop = 0.02 * max(total_range, 1e-6)
        confirm_window = 40

        for i in range(s1, min(s1 + 80, n - confirm_window - 1)):
            if yvals[i] < address_mean - local_drop:
                future_min = float(np.nanmin(yvals[i:i + confirm_window]))
                if address_mean - future_min > significant_drop:
                    backswing_start = max(i - 5, s1)
                    return s0, backswing_start, s1, address_mean

        # fallback: detect first streak of negative slope
        dy = np.diff(yvals)
        for i in range(20, min(250, len(dy) - 6)):
            if np.all(dy[i:i + 6] < -0.01 * max(total_range, 1e-6)):
                backswing_start = i
                return s0, backswing_start, s1, address_mean

        return s0, 0, s1, address_mean

    address_idx, backswing_start, address_end, address_y = detect_address_and_backswing_start(y)

    # store address idx
    df["address_idx"] = int(address_idx)
    df["backswing_start_idx"] = int(backswing_start)

    if debug:
        print("DEBUG address_idx:", address_idx, "address_end:", address_end, "address_y:", address_y)
        print("DEBUG backswing_start:", backswing_start)


# -----------------------------
    # 3) Top of backswing detection (turning point before big move)
    # Must have at least MIN_DROP Y movement to be valid backswing
    # -----------------------------
    search_window = min(90, len(y) - backswing_start - 1)
    segment = y[backswing_start : backswing_start + search_window]

    seg_range = float(np.max(segment) - np.min(segment))
    prom = 0.15 * max(seg_range, 1e-6)

    troughs, properties = find_peaks(
        -segment,
        prominence=prom,
        distance=8
    )

    # Minimum Y drop required from backswing_start to be considered valid backswing
    MIN_DROP_FROM_START = 0.2
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
    # 4) Impact detection (original style)
    # -----------------------------
    search_start = int(backswing_top)
    search_end = int(min(backswing_top + 80, n))

    downswing_y = y[search_start:search_end]
    if len(downswing_y) == 0:
        impact_raw = int(backswing_top)
    else:
        impact_relative = int(np.nanargmin(np.abs(downswing_y - address_y)))
        impact_raw = int(search_start + impact_relative)

    impact = int(min(max(impact_raw, 0), n - 1))

    df["impact_raw_idx"] = int(impact_raw)
    df["impact_idx"] = int(impact)

    if debug:
        print("DEBUG impact_raw:", impact_raw, "impact:", impact, "y_impact:", y[impact])


    # -----------------------------
    # 5) Hand path (3D uses Z, 2D uses X proxy)
    # -----------------------------
    wrist_y = df["wrist_y_smooth"].values.astype(float)
    wrist_x = df["wrist_x_smooth"].values.astype(float) if "wrist_x_smooth" in df.columns else None

    if "wrist_z_smooth" in df.columns:
        wrist_depth = df["wrist_z_smooth"].values.astype(float)
        has_3d = True
    else:
        wrist_depth = wrist_x  # 2D proxy
        has_3d = False

    df["has_3d_data"] = bool(has_3d)

    if wrist_depth is None:
        steepness_ratio = np.nan
        hand_path_label = "N/A (missing wrist_x)"
    else:
        y_change = float(abs(wrist_y[impact] - wrist_y[backswing_top]))
        d_change = float(abs(wrist_depth[impact] - wrist_depth[backswing_top]))
        steepness_ratio = (y_change / d_change) if d_change > 1e-6 else 1.0

        if club_type == "driver":
            if steepness_ratio > 1.2:
                hand_path_label = "Too steep for driver"
            elif steepness_ratio > 0.6:
                hand_path_label = "Neutral (good for driver)"
            else:
                hand_path_label = "Shallow (ideal for driver)"
        else:
            if steepness_ratio > 1.5:
                hand_path_label = "Steep (good for irons)"
            elif steepness_ratio > 0.8:
                hand_path_label = "Neutral (versatile)"
            else:
                hand_path_label = "Shallow (may lack compression)"

    df["hand_path_steepness"] = float(steepness_ratio) if steepness_ratio == steepness_ratio else np.nan
    df["hand_path_label"] = hand_path_label

    # -----------------------------
    # 6) Arm extension at impact (elbow angle)
    # -----------------------------
    needed = all(c in df.columns for c in ["shoulder_x_smooth", "shoulder_y_smooth","elbow_x_smooth", "elbow_y_smooth","wrist_x_smooth", "wrist_y_smooth"])
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
        elbow_angle = None

    # -----------------------------
    # 7) Wrist speed + timing (use fps if provided)
    # -----------------------------
    # If fps is not passed in, we assume 30fps.
    fps_used = float(fps) if fps is not None else float(df.get("fps", pd.Series([30])).iloc[0] if "fps" in df.columns else 30.0)
    if fps_used <= 0:
        fps_used = 30.0
    dt = 1.0 / fps_used

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
    # 8) Tempo
    # -----------------------------
    backswing_frames = int(max(backswing_top - backswing_start, 0))
    downswing_frames = int(max(impact - backswing_top, 1))  # avoid division by 0
    tempo_ratio = round(backswing_frames / downswing_frames, 2) if downswing_frames > 0 else 0.0

    df["backswing_frames"] = backswing_frames
    df["downswing_frames"] = downswing_frames
    df["tempo_ratio"] = tempo_ratio

    # -----------------------------
    # 9) Finish detection (use speed stabilization, not tiny y std)
    # -----------------------------
    # Finish when wrist speed stays low for a while after impact.
    try:
        # Reuse velocity if available
        if wrist_x is not None:
            # compute speed again quickly in pixels/sec (2D)
            vx = np.diff(wrist_x) / dt
            vy = np.diff(wrist_y) / dt
            speed = np.sqrt(vx * vx + vy * vy)
            post = speed[min(impact, len(speed)-1):]

            if len(post) < 20:
                finish = n - 1
            else:
                peak = float(np.nanmax(speed)) if len(speed) else 1.0
                thresh = 0.12 * max(peak, 1e-6)  # 12% of peak speed
                stable_len = 12

                finish = n - 1
                count = 0
                for i in range(len(post)):
                    if post[i] < thresh:
                        count += 1
                        if count >= stable_len:
                            finish = min(impact + i, n - 1)
                            break
                    else:
                        count = 0
        else:
            finish = n - 1
    except Exception:
        finish = n - 1

    df["finish_idx"] = int(finish)
    df["follow_through_frames"] = int(max(finish - impact, 0))

    # -----------------------------
    # 10) Overall score
    # -----------------------------
    score = 0
    max_score = 0

    # Tempo
    max_score += 25
    if 2.0 <= tempo_ratio <= 3.5:
        score += 25
    elif 1.5 <= tempo_ratio <= 4.0:
        score += 18
    else:
        score += 10

    # Arm extension
    if elbow_angle is not None:
        max_score += 25
        if elbow_angle > 155:
            score += 25
        elif elbow_angle > 140:
            score += 22
        elif elbow_angle > 125:
            score += 15
        else:
            score += 8
    else:
        max_score += 25
        score += 12  # neutral if missing

    # Hand path
    max_score += 25
    sr = df["hand_path_steepness"].iloc[0]
    if sr == sr:  # not nan
        if club_type == "driver":
            if 0.4 <= sr <= 1.0:
                score += 25
            elif 0.3 <= sr <= 1.2:
                score += 18
            else:
                score += 10
        else:
            if 0.8 <= sr <= 1.5:
                score += 25
            elif 0.5 <= sr <= 2.0:
                score += 18
            else:
                score += 10
    else:
        score += 12

    # Speed timing
    max_score += 25
    speed_timing_score = int(df["speed_timing_score"].iloc[0]) if "speed_timing_score" in df.columns else 0
    score += int(speed_timing_score * 0.25)

    overall = round(score / max_score * 100) if max_score > 0 else 0

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


def print_analysis_results(df):
    print("\n" + "=" * 60)
    print("🏌️ SWING ANALYSIS RESULTS")
    print("=" * 60)

    club = df.get("club_type", pd.Series(["iron"])).iloc[0]
    has_3d = bool(df.get("has_3d_data", pd.Series([False])).iloc[0])

    print(f"\n🏌️ Club Type: {str(club).upper()}")
    print(f"📊 Data Type: {'3D' if has_3d else '2D (MediaPipe)'}")

    def gi(name, default=0):
        return int(df[name].iloc[0]) if name in df.columns else default

    print("\n📍 PHASE DETECTION:")
    print(f"   Address:              Frame {gi('address_idx')}")
    print(f"   Backswing Start:      Frame {gi('backswing_start_idx')}")
    print(f"   Top of Backswing:     Frame {gi('backswing_top_idx')}")
    print(f"   Impact (raw):         Frame {gi('impact_raw_idx')}")
    print(f"   Impact (final):       Frame {gi('impact_idx')}")
    print(f"   Finish:               Frame {gi('finish_idx')}")

    print("\n⏱️ TEMPO:")
    print(f"   Backswing frames:  {gi('backswing_frames')}")
    print(f"   Downswing frames:  {gi('downswing_frames')}")
    print(f"   Tempo ratio:       {df['tempo_ratio'].iloc[0] if 'tempo_ratio' in df.columns else 'N/A'}:1")

    print("\n🖐️ HAND PATH:")
    if "hand_path_steepness" in df.columns:
        print(f"   Steepness ratio:   {df['hand_path_steepness'].iloc[0]:.2f}")
        print(f"   Label:             {df['hand_path_label'].iloc[0]}")
    else:
        print("   N/A")

    print("\n💪 ARM EXTENSION:")
    if "elbow_angle_impact" in df.columns and df["elbow_angle_impact"].iloc[0] == df["elbow_angle_impact"].iloc[0]:
        print(f"   Elbow angle:       {df['elbow_angle_impact'].iloc[0]:.1f}°")
        print(f"   Label:             {df['arm_extension_label'].iloc[0]}")
    else:
        print("   N/A")

    print("\n⚡ SPEED TIMING:")
    if "max_speed_frame" in df.columns:
        print(f"   Max speed frame:   {gi('max_speed_frame')}")
        print(f"   Timing:            {df['speed_timing'].iloc[0]}")
        print(f"   Diff frames:       {gi('max_speed_frames_from_impact')}")
    else:
        print("   N/A")

    print("\n🏆 OVERALL:")
    print(f"   Score:  {df['overall_score'].iloc[0] if 'overall_score' in df.columns else 'N/A'}/100")
    print(f"   Rating: {df['overall_rating'].iloc[0] if 'overall_rating' in df.columns else 'N/A'}")

    print("=" * 60)

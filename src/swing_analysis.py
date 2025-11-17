import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def analyze_swing(df):
    """
    Full swing analysis pipeline.
    """

    # -----------------------------------
    # 1. Smooth wrist coordinates
    # -----------------------------------
    df["wrist_x_smooth"] = df["wrist_x"].rolling(5, min_periods=1).mean()
    df["wrist_y_smooth"] = df["wrist_y"].rolling(5, min_periods=1).mean()
    df["wrist_z_smooth"] = df["wrist_z"].rolling(5, min_periods=1).mean()

    y = df["wrist_y_smooth"].values

    # -----------------------------------
    # 2. Detect backswing start
    # -----------------------------------
    # Detect backswing start
    def detect_backswing_start_robust(y):
        window = 20
        max_search = min(len(y)//2, 250)
        stability_threshold = 0.01
        min_stable_length = 10
        max_stable_length = 40

        rolling_std = pd.Series(y).rolling(window, center=True).std()

        stable_segments = []
        current_start = None
        current_length = 0

        for i in range(window, max_search):
            if rolling_std.iloc[i] < stability_threshold:
                if current_start is None:
                    current_start = i
                current_length += 1
            else:
                if current_start is not None and current_length >= min_stable_length:
                    stable_segments.append((current_start, i-1, current_length))
                current_start = None
                current_length = 0

        if current_start is not None and current_length >= min_stable_length:
            stable_segments.append((current_start, max_search-1, current_length))

        if stable_segments:
            stable_segments.sort(key=lambda x: x[2], reverse=True)
            stable_start, stable_end, _ = stable_segments[0]

            # ✅ Backswing starts **immediately after the stable period ends**
            backswing_start = stable_end + 1
            return backswing_start

        # fallback: derivative-based
        dy = np.diff(y)
        derivative_drop = -0.01
        consecutive = 5
        for i in range(20, min(250, len(dy)-consecutive)):
            if np.all(dy[i:i+consecutive] < derivative_drop):
                return i

        return 0


    backswing_start = detect_backswing_start_robust(y)

    
    # -----------------------------------
    # 3. Top of backswing (lowest Y after start)
    # -----------------------------------
    search_end = min(backswing_start + 80, len(df) - 1)
    segment = y[backswing_start:search_end]
    
    # Compute frame-to-frame difference
    dy = np.diff(segment)
    
    # Find first point where derivative changes from negative to positive
    turning_points = np.where(dy > 0)[0]
    if len(turning_points) > 0:
        backswing_top = backswing_start + turning_points[0]
    else:
    # fallback: lowest point if no turning point found
        backswing_top = backswing_start + np.argmin(segment)
    

    # -----------------------------------
    # 4. Impact detection (highest velocity)
    # -----------------------------------
    impact_search_end = min(backswing_top + 50, len(y) - 1)
    segment_downswing = y[backswing_top:impact_search_end]
    
    peaks, _ = find_peaks(segment_downswing)

    if len(peaks) > 0:
        impact = backswing_top + peaks[np.argmax(segment_downswing[peaks])]
    else:
        impact = backswing_top + len(segment_downswing) - 1

    # Save indices
    df["backswing_start_idx"] = backswing_start
    df["backswing_top_idx"] = backswing_top
    df["impact_idx"] = impact

    # -----------------------------------
    # 5. Swing Path (X vs Z)
    # -----------------------------------
    # -----------------------------------
    # Use smoothed X values
    x = df["wrist_x_smooth"].values

    # Pre- and post-impact frames
    pre_frame  = max(impact - 8, 0)
    post_frame = min(impact + 8, len(x) - 1)

    # Compute delta in both directions
    delta_x = x[post_frame] - x[pre_frame]

    # Convert delta to angle (scaled)
    swing_angle = np.degrees(np.arctan(delta_x * 6))  # scale factor 6, slightly softer

    # Classification
    if delta_x > 0.02:
        label = "In-to-Out (Draw/Hook tendency)"
    elif delta_x < -0.02:
        label = "Out-to-In (Fade/Slice tendency)"
    else:
        label = "Neutral / Straight"

    df["swing_path_angle"] = swing_angle
    df["swing_path_label"] = label
    df["path_quality"] = 1.0


    # -----------------------------------
    # 6. Ball flight classification
    # -----------------------------------
    abs_angle = abs(swing_angle)

    if abs_angle < 2:
        label = "Neutral / Straight"
        ball_flight = "Straight"
    elif abs_angle < 5:
        label = "Slight In-to-Out" if swing_angle > 0 else "Slight Out-to-In"
        ball_flight = "Baby Draw" if swing_angle > 0 else "Baby Fade"
    elif abs_angle < 8:
        label = "In-to-Out (Draw)" if swing_angle > 0 else "Out-to-In (Fade)"
        ball_flight = "Draw" if swing_angle > 0 else "Fade"
    else:
        label = "Strong In-to-Out (Hook)" if swing_angle > 0 else "Strong Out-to-In (Slice)"
        ball_flight = "Hook" if swing_angle > 0 else "Slice"

    # -----------------------------------
    # 7. Tempo calculation
    # -----------------------------------
    backswing_time = backswing_top - backswing_start
    downswing_time = impact - backswing_top

    tempo_ratio = round(backswing_time / downswing_time, 2) if downswing_time > 0 else 0.0

    
    df["ball_flight"] = ball_flight
    df["tempo_ratio"] = tempo_ratio

    df["backswing_frames"] = backswing_time
    df["downswing_frames"] = downswing_time

    return df

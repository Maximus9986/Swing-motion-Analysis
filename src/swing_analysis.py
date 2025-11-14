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
    def detect_backswing_start_robust(y):
        """
        Robust backswing detection that adapts to different videos.
        Strategy: Find long stable period, then detect significant deviation.
        """
        
        # Calculate rolling statistics
        window = 20
        rolling_mean = pd.Series(y).rolling(window, center=True).mean()
        rolling_std = pd.Series(y).rolling(window, center=True).std()
        
        # Find the longest stable period in the first half of video
        max_search = min(len(y) // 2, 250)  # Don't search past halfway point or frame 250
        
        stability_threshold = 0.01  # Low std = stable
        min_stable_length = 40  # Need at least 40 frames of stability
        
        stable_start = None
        stable_end = None
        longest_stable = 0
        current_stable_start = None
        current_stable_length = 0
        
        # Find longest stable segment
        for i in range(window, max_search):
            if rolling_std.iloc[i] < stability_threshold:
                if current_stable_start is None:
                    current_stable_start = i
                current_stable_length += 1
            else:
                # Stability broken
                if current_stable_length > longest_stable:
                    longest_stable = current_stable_length
                    stable_start = current_stable_start
                    stable_end = i - 1
                current_stable_start = None
                current_stable_length = 0
        
        # Check final segment
        if current_stable_length > longest_stable:
            stable_start = current_stable_start
            stable_end = max_search - 1
        
        # If we found a stable period, look for movement after it
        if stable_start is not None and (stable_end - stable_start) >= min_stable_length:
            # Get the mean Y value during stable address
            address_mean = np.mean(y[stable_start:stable_end])
            
            # Look for significant drop after stable period
            drop_threshold = 0.02  # 2cm drop from address mean
            
            for i in range(stable_end, min(stable_end + 50, len(y))):
                if y[i] < address_mean - drop_threshold:
                    return max(i - 5, stable_end)  # Go back 5 frames to catch start of movement
        
        # Fallback: use derivative-based detection
        dy = np.diff(y)
        drop_threshold = -0.01
        consecutive = 5
        
        for i in range(20, min(250, len(dy) - consecutive)):
            if np.all(dy[i:i + consecutive] < drop_threshold):
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

    # Look 8 frames before impact (or as many as available)
    pre_impact_frame = max(impact - 8, 0)

    # Change in X direction right before impact
    delta_x = x[impact] - x[pre_impact_frame]

    # Swing path angle (simple proxy): arctan(delta_x / small forward-distance)
    # We use a small constant scale to convert delta_x into an angle-like value
    swing_angle = np.degrees(np.arctan(delta_x * 8))  # scale factor = 8 (tunable)

    # Path quality placeholder for now
    path_quality = 1.0

    # Classify ball flight based on delta_x
    if delta_x > 0.02:
        label = "In-to-Out (Draw/Hook tendency)"
    elif delta_x < -0.02:
        label = "Out-to-In (Fade/Slice tendency)"
    else:
        label = "Neutral / Straight"

    df["swing_path_angle"] = swing_angle
    df["swing_path_label"] = label
    df["path_quality"] = path_quality


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

    df["swing_path_angle"] = swing_angle
    df["swing_path_label"] = label
    df["ball_flight"] = ball_flight
    df["path_quality"] = path_quality
    df["tempo_ratio"] = tempo_ratio

    df["backswing_frames"] = backswing_time
    df["downswing_frames"] = downswing_time

    return df

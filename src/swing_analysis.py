import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress


def analyze_swing(df):
    """
    Full swing analysis pipeline with robust phase detection.
    Returns analyzed DataFrame with swing metrics.
    """
    
    if df is None or df.empty:
        return pd.DataFrame()
    
    try:
        # -----------------------------------
        # 1. Smooth wrist coordinates
        # -----------------------------------
        for joint in ["wrist", "elbow", "hip"]:
            for axis in ["x", "y", "z"]:
                col = f"{joint}_{axis}"
                if col in df.columns:
                    # Use Savitzky-Golay filter for smoothing
                    df[f"{col}_smooth"] = savgol_filter(
                        df[col].fillna(method='ffill').fillna(method='bfill'), 
                        window_length=7, 
                        polyorder=2, 
                        mode='nearest'
                    )

        y = df["wrist_y_smooth"].values

        # -----------------------------------
        # 2. Detect backswing start - ROBUST VERSION
        # -----------------------------------
        def detect_backswing_start_robust(y):
            y = np.asarray(y)

            # -------------------------------
            # 1. Compute global movement scale
            # -------------------------------
            total_range = np.percentile(y, 95) - np.percentile(y, 5)
            significant_drop = 0.30 * total_range   # 30% of full swing height

            # -------------------------------
            # 2. Detect address stability
            # -------------------------------
            window = 20
            rolling_std = (
                pd.Series(y)
                .rolling(window, center=True)
                .std()
                .bfill()
                .ffill()
            )

            max_search = min(len(y) // 2, 250)
            stability_threshold = 0.06
            min_stable_length = 12

            stable_segments = []
            current_start = None
            current_len = 0

            for i in range(window, max_search):
                if rolling_std.iloc[i] < stability_threshold:
                    if current_start is None:
                        current_start = i
                    current_len += 1
                else:
                    if current_start is not None and current_len >= min_stable_length:
                        stable_segments.append((current_start, i - 1, current_len))
                    current_start = None
                    current_len = 0

            if current_start is not None and current_len >= min_stable_length:
                stable_segments.append((current_start, max_search - 1, current_len))

            # -------------------------------
            # 3. Pick best address segment
            # -------------------------------
            if stable_segments:
                stable_segments.sort(key=lambda x: x[2], reverse=True)
                stable_start, stable_end, _ = stable_segments[0]
                address_mean = np.mean(y[stable_start:stable_end])
            else:
                stable_end = 20
                address_mean = np.mean(y[:20])

            # -------------------------------
            # 4. Detect true backswing start
            # -------------------------------
            local_drop = 0.02
            confirm_window = 40

            for i in range(stable_end, min(stable_end + 60, len(y) - confirm_window)):
                if y[i] < address_mean - local_drop:
                    future_min = np.min(y[i:i + confirm_window])

                    if address_mean - future_min > significant_drop:
                        return max(i - 5, stable_end)

            # -------------------------------
            # 5. Fallback: derivative-based
            # -------------------------------
            dy = np.diff(y)
            for i in range(20, min(250, len(dy) - 6)):
                if np.all(dy[i:i + 6] < -0.01):
                    return i

            return 0


        backswing_start = detect_backswing_start_robust(y)

        # -----------------------------------
        # 3. Top of backswing 
        # -----------------------------------

        search_window = min(90, len(y) - backswing_start - 1)
        segment = y[backswing_start: backswing_start + search_window]

        # Find troughs by finding peaks in -Y
        troughs, properties = find_peaks(
            -segment,
            prominence=0.15 * (np.max(segment) - np.min(segment)),
            distance=8
        )

        if len(troughs) > 0:
            backswing_top = backswing_start + troughs[0]
        else:
            # Fallback: absolute minimum
            backswing_top = backswing_start + np.argmin(segment)

        # Debug
        print(f"DEBUG: Backswing start = {backswing_start}")
        print(f"DEBUG: Backswing top = {backswing_top}")
        print(f"DEBUG: Y at top = {y[backswing_top]:.4f}")

        # -----------------------------------
        # 4. Impact detection - FIXED VERSION
        # Find where wrist Y returns to address level
        # -----------------------------------
        
        # Get address Y level (average Y around backswing start)
        address_y = np.mean(y[max(0, backswing_start-5):backswing_start+5])
        print(f"DEBUG: Address Y level = {address_y:.4f}")
        
        # Search window after top
        search_start = backswing_top
        search_end = min(backswing_top + 40, len(y))
        
        # Find frame where Y is closest to address_y
        downswing_y = y[search_start:search_end]
        impact_relative = np.argmin(np.abs(downswing_y - address_y))
        impact = search_start + impact_relative
        

        # Save indices
        df.loc[:, "backswing_start_idx"] = backswing_start
        df.loc[:, "backswing_top_idx"] = backswing_top
        df.loc[:, "impact_idx"] = impact

        # -----------------------------------
        # 5. Swing Path (X vs Z) - WRIST PATH
        # -----------------------------------
        downswing_frames = impact - backswing_top
        
        # Adaptive window based on frame rate
        if downswing_frames < 10:
            pre_impact_frames = 3
            post_impact_frames = 2
        elif downswing_frames < 20:
            pre_impact_frames = 5
            post_impact_frames = 3
        else:
            pre_impact_frames = 8
            post_impact_frames = 5
        
        start_frame = max(impact - pre_impact_frames, backswing_top)
        end_frame = min(impact + post_impact_frames, len(df) - 1)
        
        num_samples = min(max(6, (end_frame - start_frame)), 12)
        sample_frames = np.linspace(start_frame, end_frame, num_samples, dtype=int)

        x_vals = df.loc[sample_frames, "wrist_x_smooth"].values
        z_vals = df.loc[sample_frames, "wrist_z_smooth"].values

        mask = ~(np.isnan(x_vals) | np.isnan(z_vals))
        x_vals, z_vals = x_vals[mask], z_vals[mask]

        if len(x_vals) >= 5:
            slope, _, r_val, _, _ = linregress(z_vals, x_vals)
            swing_angle = np.degrees(np.arctan(slope))
            path_quality = r_val**2
            
            # Conservative thresholds for wrist path
            if swing_angle > 1.5:
                label = "In-to-Out (Draw/Hook tendency)"
            elif swing_angle < -1.5:
                label = "Out-to-In (Fade/Slice tendency)"
            else:
                label = "Neutral / Straight"
        else:
            swing_angle = 0.0
            path_quality = 0.0
            label = "Neutral / Straight"

        df["swing_path_angle"] = swing_angle
        df["swing_path_label"] = label
        df["path_quality"] = path_quality

        # -----------------------------------
        # 6. Ball flight classification
        # -----------------------------------
        abs_angle = abs(swing_angle)
        if abs_angle < 1:
            ball_flight = "Straight"
        elif abs_angle < 3:
            ball_flight = "Baby Draw" if swing_angle > 0 else "Baby Fade"
        elif abs_angle < 5:
            ball_flight = "Draw" if swing_angle > 0 else "Fade"
        else:
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

        # -----------------------------------
        # 8. Finish detection
        # -----------------------------------
        if impact + 20 < len(y):
            post_impact_std = pd.Series(y[impact:]).rolling(10).std()
            stable_frames = np.where(post_impact_std < 0.02)[0]
            if len(stable_frames) > 0:
                finish = impact + stable_frames[0]
            else:
                finish = len(y) - 1
        else:
            finish = len(y) - 1
        
        df.loc[:, "finish_idx"] = finish
        df["follow_through_frames"] = finish - impact

        return df
        
    except Exception as e:
        print(f"Swing analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return df  # Return original df instead of empty
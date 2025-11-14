import numpy as np
import pandas as pd


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
    #    First major drop in wrist height
    # -----------------------------------
    dy = np.diff(y)
    drop_threshold = -0.015
    drop_indices = np.where(dy < drop_threshold)[0]

    backswing_start = drop_indices[0] if len(drop_indices) else 0

    # -----------------------------------
    # 3. Top of backswing (lowest Y after start)
    # -----------------------------------
    search_end = min(backswing_start + 80, len(df) - 1)
    segment = y[backswing_start:search_end]
    backswing_top = backswing_start + np.argmin(segment)

    # -----------------------------------
    # 4. Impact detection (highest velocity)
    # -----------------------------------
    vel = np.abs(np.diff(y))
    impact_search_start = backswing_top
    impact_search_end = min(impact_search_start + 40, len(vel) - 1)

    if impact_search_end <= impact_search_start:
        impact = backswing_top + 10
    else:
        local_vel = vel[impact_search_start:impact_search_end]
        impact = impact_search_start + np.argmax(local_vel)

    # Save indices
    df["backswing_start_idx"] = backswing_start
    df["backswing_top_idx"] = backswing_top
    df["impact_idx"] = impact

    # -----------------------------------
    # 5. Swing Path (X vs Z)
    # -----------------------------------
    try:
        from scipy.stats import linregress

        sample_frames = np.linspace(backswing_top, impact, 10, dtype=int)

        x_vals = df.loc[sample_frames, "wrist_x_smooth"].values
        z_vals = df.loc[sample_frames, "wrist_z_smooth"].values

        mask = ~(np.isnan(x_vals) | np.isnan(z_vals))
        x_vals, z_vals = x_vals[mask], z_vals[mask]

        if len(x_vals) >= 3:
            slope, _, r_val, _, _ = linregress(z_vals, x_vals)
            swing_angle = np.degrees(np.arctan(slope))
            path_quality = r_val**2
        else:
            swing_angle = 0.0
            path_quality = 0.0

    except Exception:
        swing_angle = 0.0
        path_quality = 0.0

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

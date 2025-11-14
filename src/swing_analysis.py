import numpy as np
import pandas as pd

def analyze_swing(df):
    """
    Analyze golf swing from wrist tracking data.
    
    Args:
        df: DataFrame with columns 'x', 'y', 'z', 'frame'
        
    Returns:
        DataFrame with analysis results
    """
    
    # Smooth wrist coordinates
    df["wrist_x_smooth"] = df["x"].rolling(window=5, min_periods=1).mean()
    df["wrist_y_smooth"] = df["y"].rolling(window=5, min_periods=1).mean()
    df["wrist_z_smooth"] = df["z"].rolling(window=5, min_periods=1).mean()
    
    # Calculate wrist speed
    df["wrist_velocity_y"] = df["wrist_y_smooth"].diff()
    df["wrist_speed"] = df["wrist_velocity_y"].abs()
    
    # Find backswing peak (lowest Y point)
    downswing_start = df["wrist_y_smooth"].idxmin()
    
    # Find impact (speed peak after backswing)
    search_end = min(downswing_start + 20, len(df) - 1)
    search_window = df.loc[downswing_start:search_end]
    
    if len(search_window) > 0:
        impact_frame = search_window["wrist_speed"].idxmax()
    else:
        impact_frame = downswing_start + 10
    
    print(f"[INFO] Backswing peak: frame {downswing_start}")
    print(f"[INFO] Impact: frame {impact_frame}")
    
    # Calculate swing path
    try:
        from scipy.stats import linregress
        
        # Get sample points between backswing and impact
        num_points = min(10, impact_frame - downswing_start)
        
        if num_points >= 3:
            sample_frames = np.linspace(downswing_start, impact_frame, num_points, dtype=int)
            
            x_coords = df.loc[sample_frames, "wrist_x_smooth"].values
            z_coords = df.loc[sample_frames, "wrist_z_smooth"].values
            
            # Remove NaN
            valid_mask = ~(np.isnan(x_coords) | np.isnan(z_coords))
            x_coords = x_coords[valid_mask]
            z_coords = z_coords[valid_mask]
            
            if len(x_coords) >= 3:
                # Simple angle
                lateral = x_coords[-1] - x_coords[0]
                forward = abs(z_coords[-1] - z_coords[0])
                if forward < 0.001:
                    forward = 0.001
                raw_angle = np.degrees(np.arctan(lateral / forward))
                
                # Regression
                slope, intercept, r_value, p_value, std_err = linregress(z_coords, x_coords)
                regression_angle = np.degrees(np.arctan(slope))
                path_quality = r_value ** 2
                
                # Average both methods
                swing_angle = (raw_angle + regression_angle) / 2
            else:
                swing_angle = 0.0
                path_quality = 0.0
        else:
            swing_angle = 0.0
            path_quality = 0.0
            
    except Exception as e:
        print(f"[Warning] {e}")
        swing_angle = 0.0
        path_quality = 0.0
    
    # Classify swing path
    abs_angle = abs(swing_angle)
    
    if abs_angle < 2:
        label = "Neutral / Straight"
        ball_flight = "Straight"
    elif abs_angle < 5:
        if swing_angle > 0:
            label = "Slight In-to-Out"
            ball_flight = "Baby Draw"
        else:
            label = "Slight Out-to-In"
            ball_flight = "Baby Fade"
    elif abs_angle < 8:
        if swing_angle > 0:
            label = "In-to-Out (Draw)"
            ball_flight = "Draw"
        else:
            label = "Out-to-In (Fade)"
            ball_flight = "Fade"
    else:
        if swing_angle > 0:
            label = "Strong In-to-Out (Hook)"
            ball_flight = "Hook"
        else:
            label = "Strong Out-to-In (Slice)"
            ball_flight = "Slice"
    
    # Store results
    df["swing_path_angle"] = swing_angle
    df["swing_path_label"] = label
    df["ball_flight"] = ball_flight
    df["path_quality"] = path_quality
    
    print(f"[INFO] Swing path: {label} ({swing_angle:.2f}°)")
    print(f"[INFO] Quality R²: {path_quality:.3f}")
    
    return df
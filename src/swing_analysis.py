import numpy as np
import pandas as pd

def analyze_swing(df):
    """
    Analyze golf swing data using velocity-based phase detection.
    
    Args:
        df: DataFrame with columns 'x', 'y', 'z', 'frame' from pose_tracking
        
    Returns:
        DataFrame with added analysis columns
    """
    
    # Check if we have the required columns
    if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
        raise ValueError(f"Missing required columns. Found columns: {df.columns.tolist()}")
    
    # Smooth wrist coordinates
    df["wrist_x_smooth"] = df["x"].rolling(window=5, min_periods=1).mean()
    df["wrist_y_smooth"] = df["y"].rolling(window=5, min_periods=1).mean()
    df["wrist_z_smooth"] = df["z"].rolling(window=5, min_periods=1).mean()

    # Calculate wrist velocity (change in Y position)
    df["wrist_velocity_y"] = df["wrist_y_smooth"].diff()
    
    # Calculate wrist speed (absolute velocity)
    df["wrist_speed"] = df["wrist_velocity_y"].abs()

    # ✅ IMPROVED: Find backswing peak using velocity change
    # The top of backswing is where velocity changes from positive to negative
    # (wrist stops going up and starts coming down)
    
    # Find where velocity crosses from positive to negative
    df["velocity_sign"] = np.sign(df["wrist_velocity_y"])
    df["velocity_change"] = df["velocity_sign"].diff()
    
    # Backswing peak is where velocity_change is negative (going from up to down)
    # Look in the first half of the video where backswing typically happens
    search_region = df.iloc[10:len(df)//2]  # Skip first 10 frames, search first half
    
    # Find where wrist stops going up (velocity changes from + to -)
    velocity_changes = search_region[search_region["velocity_change"] < 0]
    
    if len(velocity_changes) > 0:
        # The first significant direction change is likely the top of backswing
        downswing_start = velocity_changes.index[0]
        print(f"[DEBUG] Found backswing peak at frame {downswing_start} using velocity change")
    else:
        # Fallback: use highest point
        downswing_start = df["wrist_y_smooth"].idxmax()
        print(f"[DEBUG] Using highest point fallback at frame {downswing_start}")
    
    # Find impact using speed peak after backswing
    search_end = min(downswing_start + 20, len(df) - 1)
    search_window = df.loc[downswing_start:search_end]
    
    if len(search_window) > 0 and "wrist_speed" in search_window.columns:
        impact_frame = search_window["wrist_speed"].idxmax()
        if pd.isna(impact_frame):
            impact_frame = downswing_start + 10
    else:
        impact_frame = downswing_start + 10
    
    print(f"[DEBUG] Backswing peak (transition): frame {downswing_start}")
    print(f"[DEBUG] Impact point: frame {impact_frame}")

    # Calculate swing path
    try:
        from scipy.stats import linregress
        
        # Get sample points from downswing to impact
        num_points = min(10, impact_frame - downswing_start)
        if num_points >= 5:
            sample_frames = np.linspace(downswing_start, impact_frame, num_points, dtype=int)
            sample_frames = sample_frames[sample_frames < len(df)]
            
            x_coords = df.loc[sample_frames, "wrist_x_smooth"].values
            z_coords = df.loc[sample_frames, "wrist_z_smooth"].values
            y_coords = df.loc[sample_frames, "wrist_y_smooth"].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(x_coords) | np.isnan(z_coords) | np.isnan(y_coords))
            x_coords = x_coords[valid_mask]
            z_coords = z_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            
            if len(x_coords) >= 3:
                # Calculate lateral and forward movement
                total_movement = np.sqrt(
                    (x_coords[-1] - x_coords[0])**2 +
                    (z_coords[-1] - z_coords[0])**2
                )
                
                if total_movement < 0.001:
                    swing_angle = 0.0
                    path_quality = 1.0
                else:
                    lateral_movement = x_coords[-1] - x_coords[0]
                    forward_movement = abs(z_coords[-1] - z_coords[0])
                    
                    if forward_movement < 0.001:
                        forward_movement = 0.001
                    
                    # Simple angle calculation
                    raw_angle = np.degrees(np.arctan(lateral_movement / forward_movement))
                    
                    # Linear regression for quality
                    slope, intercept, r_value, p_value, std_err = linregress(z_coords, x_coords)
                    regression_angle = np.degrees(np.arctan(slope))
                    path_quality = r_value ** 2
                    
                    # Average the two methods
                    swing_angle = (raw_angle + regression_angle) / 2
                    
                    print(f"[DEBUG] Lateral movement: {lateral_movement:.4f}")
                    print(f"[DEBUG] Forward movement: {forward_movement:.4f}")
                    print(f"[DEBUG] Calculated angle: {swing_angle:.2f}°")
            else:
                swing_angle = 0.0
                path_quality = 0.0
        else:
            swing_angle = 0.0
            path_quality = 0.0
    
    except ImportError:
        print("[INFO] scipy not available")
        x_start = df.loc[downswing_start, "wrist_x_smooth"]
        x_end = df.loc[impact_frame, "wrist_x_smooth"]
        z_start = df.loc[downswing_start, "wrist_z_smooth"]
        z_end = df.loc[impact_frame, "wrist_z_smooth"]
        
        lateral = x_end - x_start
        forward = abs(z_end - z_start)
        if forward < 0.001:
            forward = 0.001
        swing_angle = np.degrees(np.arctan(lateral / forward))
        path_quality = 0.0
    
    except Exception as e:
        print(f"[Warning] Swing path calculation error: {e}")
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

    # Store results AND the correct backswing peak frame
    df["swing_path_angle"] = swing_angle
    df["swing_path_label"] = label
    df["ball_flight"] = ball_flight
    df["path_quality"] = path_quality
    df["backswing_peak_frame"] = downswing_start  # ✅ Store the correct frame!

    print(f"[INFO] Swing Analysis:")
    print(f"  - Backswing peak: frame {downswing_start}")
    print(f"  - Impact: frame {impact_frame}")
    print(f"  - Angle: {swing_angle:.2f}°")
    print(f"  - Path: {label}")
    print(f"  - Expected flight: {ball_flight}")
    print(f"  - Quality R²: {path_quality:.3f}")

    return df
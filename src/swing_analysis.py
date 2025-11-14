"""
Improved Swing Analysis - Accounts for Body Rotation
Calculates swing path relative to shoulder line, not absolute coordinates
"""

import numpy as np
import pandas as pd
import math

def analyze_swing(df):
    """
    Analyze golf swing using full body tracking.
    Calculates swing path RELATIVE to body rotation for accuracy.
    
    Args:
        df: DataFrame with full body pose data
        
    Returns:
        DataFrame with analysis results
    """
    
    print("\n[INFO] ===== FULL BODY SWING ANALYSIS =====")
    
    # Calculate shoulder line angle (body rotation)
    df["shoulder_angle"] = df.apply(calculate_shoulder_angle, axis=1)
    
    # Calculate hip line angle (lower body rotation)
    df["hip_angle"] = df.apply(calculate_hip_angle, axis=1)
    
    # Use lead wrist (left for right-handed golfer)
    # TODO: Add handedness detection or let user specify
    df["wrist_x"] = df["left_wrist_x"]
    df["wrist_y"] = df["left_wrist_y"]
    df["wrist_z"] = df["left_wrist_z"]
    
    # Smooth wrist coordinates
    df["wrist_x_smooth"] = df["wrist_x"].rolling(window=5, min_periods=1).mean()
    df["wrist_y_smooth"] = df["wrist_y"].rolling(window=5, min_periods=1).mean()
    df["wrist_z_smooth"] = df["wrist_z"].rolling(window=5, min_periods=1).mean()
    
    # Calculate wrist velocity and speed
    df["wrist_velocity_y"] = df["wrist_y_smooth"].diff()
    df["wrist_speed"] = df["wrist_velocity_y"].abs()
    
    # ✅ STEP 1: Find IMPACT (biggest speed spike)
    search_region = df.iloc[10:-10] if len(df) > 20 else df
    if len(search_region) > 0 and "wrist_speed" in search_region.columns:
        impact_frame = search_region["wrist_speed"].idxmax()
        print(f"[DEBUG] Impact detected at frame {impact_frame}")
    else:
        impact_frame = len(df) // 2
        print(f"[DEBUG] Using estimated impact at frame {impact_frame}")
    
    # ✅ STEP 2: Find BACKSWING PEAK (lowest wrist Y before impact)
    backswing_search = df.iloc[10:impact_frame] if impact_frame > 10 else df.iloc[:impact_frame]
    
    if len(backswing_search) > 5:
        # Find lowest wrist point (top of backswing)
        downswing_start = backswing_search["wrist_y_smooth"].idxmin()
        print(f"[DEBUG] Backswing peak at frame {downswing_start}")
    else:
        downswing_start = max(0, impact_frame - 15)
        print(f"[DEBUG] Using estimated backswing at frame {downswing_start}")
    
    print(f"[DEBUG] Analysis window: frames {downswing_start} to {impact_frame}")
    
    # ✅ STEP 3: Calculate CORRECTED swing path (relative to body rotation)
    try:
        from scipy.stats import linregress
        
        # Get sample points between backswing and impact
        num_frames = impact_frame - downswing_start
        num_points = min(10, num_frames)
        
        if num_points >= 3:
            sample_frames = np.linspace(downswing_start, impact_frame, num_points, dtype=int)
            sample_frames = sample_frames[sample_frames < len(df)]
            
            # Get wrist coordinates
            wrist_x = df.loc[sample_frames, "wrist_x_smooth"].values
            wrist_z = df.loc[sample_frames, "wrist_z_smooth"].values
            
            # Get shoulder rotation at these frames
            shoulder_angles = df.loc[sample_frames, "shoulder_angle"].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(wrist_x) | np.isnan(wrist_z) | np.isnan(shoulder_angles))
            wrist_x = wrist_x[valid_mask]
            wrist_z = wrist_z[valid_mask]
            shoulder_angles = shoulder_angles[valid_mask]
            
            if len(wrist_x) >= 3:
                # Method 1: Simple wrist movement
                wrist_lateral = wrist_x[-1] - wrist_x[0]
                wrist_forward = abs(wrist_z[-1] - wrist_z[0])
                
                if wrist_forward < 0.001:
                    wrist_forward = 0.001
                
                wrist_angle = np.degrees(np.arctan(wrist_lateral / wrist_forward))
                
                # Method 2: Account for body rotation
                # Calculate how much shoulder line rotated
                shoulder_rotation = shoulder_angles[-1] - shoulder_angles[0]
                
                # Correct swing path by removing body rotation component
                corrected_angle = wrist_angle - (shoulder_rotation * 0.5)  # 0.5 factor since arms move relative to shoulders
                
                # Method 3: Linear regression for quality
                slope, intercept, r_value, p_value, std_err = linregress(wrist_z, wrist_x)
                regression_angle = np.degrees(np.arctan(slope))
                path_quality = r_value ** 2
                
                # Final angle: average of corrected method and regression
                swing_angle = (corrected_angle + regression_angle) / 2
                
                print(f"\n[DEBUG] Swing path calculation:")
                print(f"  - Wrist movement angle: {wrist_angle:.2f}°")
                print(f"  - Shoulder rotation: {shoulder_rotation:.2f}°")
                print(f"  - Corrected angle: {corrected_angle:.2f}°")
                print(f"  - Regression angle: {regression_angle:.2f}°")
                print(f"  - FINAL angle: {swing_angle:.2f}°")
                print(f"  - Path quality R²: {path_quality:.3f}")
                
            else:
                swing_angle = 0.0
                path_quality = 0.0
                print("[DEBUG] Not enough valid points")
        else:
            swing_angle = 0.0
            path_quality = 0.0
            print("[DEBUG] Not enough frames between backswing and impact")
    
    except ImportError:
        print("[INFO] scipy not available, using simple calculation")
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
        print(f"[Warning] Swing path error: {e}")
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
    df["backswing_peak_frame"] = downswing_start
    df["impact_frame"] = impact_frame
    
    # Also keep for compatibility with old code
    df["x"] = df["wrist_x"]
    df["y"] = df["wrist_y"]
    df["z"] = df["wrist_z"]
    
    print(f"\n[INFO] ===== FINAL RESULTS =====")
    print(f"  - Backswing peak: frame {downswing_start}")
    print(f"  - Impact: frame {impact_frame}")
    print(f"  - Swing path: {label}")
    print(f"  - Angle: {swing_angle:.2f}°")
    print(f"  - Ball flight: {ball_flight}")
    print(f"  - Quality R²: {path_quality:.3f}")
    
    return df


def calculate_shoulder_angle(row):
    """Calculate shoulder line angle (body rotation indicator)"""
    try:
        left_x = row['left_shoulder_x']
        left_z = row['left_shoulder_z']
        right_x = row['right_shoulder_x']
        right_z = row['right_shoulder_z']
        
        if pd.isna([left_x, left_z, right_x, right_z]).any():
            return np.nan
        
        # Calculate angle of shoulder line
        dx = right_x - left_x
        dz = right_z - left_z
        angle = np.degrees(np.arctan2(dx, dz))
        return angle
    except:
        return np.nan


def calculate_hip_angle(row):
    """Calculate hip line angle (lower body rotation indicator)"""
    try:
        left_x = row['left_hip_x']
        left_z = row['left_hip_z']
        right_x = row['right_hip_x']
        right_z = row['right_hip_z']
        
        if pd.isna([left_x, left_z, right_x, right_z]).any():
            return np.nan
        
        # Calculate angle of hip line
        dx = right_x - left_x
        dz = right_z - left_z
        angle = np.degrees(np.arctan2(dx, dz))
        return angle
    except:
        return np.nan
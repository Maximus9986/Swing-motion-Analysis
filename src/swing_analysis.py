import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


def analyze_swing(df, club_type="iron"):
    """
    Full swing analysis pipeline with robust phase detection.
    Works with both MediaPipe 2D and CoMotion 3D data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Pose data with wrist, elbow, shoulder, hip coordinates
    club_type : str
        "iron" or "driver" - affects impact detection
    """
    
    if df is None or df.empty:
        return pd.DataFrame()
    
    try:
        # Store club type
        df["club_type"] = club_type
        
        # -----------------------------------
        # 1. Smooth coordinates
        # -----------------------------------
        for joint in ["wrist", "elbow", "shoulder", "hip"]:
            for axis in ["x", "y", "z"]:
                col = f"{joint}_{axis}"
                if col in df.columns:
                    df[f"{col}_smooth"] = savgol_filter(
                        df[col].fillna(method='ffill').fillna(method='bfill'), 
                        window_length=7, 
                        polyorder=2, 
                        mode='nearest'
                    )

        y = df["wrist_y_smooth"].values

        # -----------------------------------
        # 2. Detect backswing start
        # -----------------------------------
        def detect_backswing_start_robust(y):
            y = np.asarray(y)

            total_range = np.percentile(y, 95) - np.percentile(y, 5)
            significant_drop = 0.30 * total_range

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

            if stable_segments:
                stable_segments.sort(key=lambda x: x[2], reverse=True)
                stable_start, stable_end, _ = stable_segments[0]
                address_mean = np.mean(y[stable_start:stable_end])
            else:
                stable_end = 20
                address_mean = np.mean(y[:20])

            local_drop = 0.02
            confirm_window = 40

            for i in range(stable_end, min(stable_end + 60, len(y) - confirm_window)):
                if y[i] < address_mean - local_drop:
                    future_min = np.min(y[i:i + confirm_window])

                    if address_mean - future_min > significant_drop:
                        return max(i - 5, stable_end)

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

        troughs, properties = find_peaks(
            -segment,
            prominence=0.15 * (np.max(segment) - np.min(segment)),
            distance=8
        )

        if len(troughs) > 0:
            backswing_top = backswing_start + troughs[0]
        else:
            backswing_top = backswing_start + np.argmin(segment)

        print(f"DEBUG: Backswing start = {backswing_start}")
        print(f"DEBUG: Backswing top = {backswing_top}")
        print(f"DEBUG: Y at top = {y[backswing_top]:.4f}")

        # -----------------------------------
        # 4. Impact detection
        # Find where wrist Y returns to address level
        # -----------------------------------
        address_y = np.mean(y[max(0, backswing_start-5):backswing_start+5])
        print(f"DEBUG: Address Y level = {address_y:.4f}")
        
        search_start = backswing_top
        search_end = min(backswing_top + 40, len(y))
        
        downswing_y = y[search_start:search_end]
        impact_relative = np.argmin(np.abs(downswing_y - address_y))
        impact_raw = search_start + impact_relative
        
        # Add offset based on club type
        if club_type.lower() == "driver":
            
            impact_offset = 0
        else:
            impact_offset = 1
        
        impact = min(impact_raw + impact_offset, len(y) - 1)
        
        print(f"DEBUG: Impact raw (address level) = {impact_raw}")
        print(f"DEBUG: Club type = {club_type}")
        print(f"DEBUG: Impact offset = {impact_offset}")
        print(f"DEBUG: Final impact = {impact}")
        print(f"DEBUG: Y at impact = {y[impact]:.4f}")

        df.loc[:, "backswing_start_idx"] = backswing_start
        df.loc[:, "backswing_top_idx"] = backswing_top
        df.loc[:, "impact_idx"] = impact

        # -----------------------------------
        # 5. Hand Path Analysis (Steep vs Shallow)
        # -----------------------------------
        wrist_y = df["wrist_y_smooth"].values
        
        # Check if Z data is available (3D) or use X as fallback (2D)
        if "wrist_z_smooth" in df.columns:
            wrist_z = df["wrist_z_smooth"].values
            has_3d = True
        else:
            # For 2D, use X as proxy for forward movement
            wrist_z = df["wrist_x_smooth"].values
            has_3d = False
        
        # Y change from top to impact
        y_change = abs(wrist_y[impact] - wrist_y[backswing_top])
        z_change = abs(wrist_z[impact] - wrist_z[backswing_top])

        if z_change > 0.001:
            steepness_ratio = y_change / z_change
        else:
            steepness_ratio = 1.0

        # Classify hand path
        if club_type.lower() == "driver":
            # Driver should be shallower
            if steepness_ratio > 1.2:
                hand_path_label = "Too steep for driver"
            elif steepness_ratio > 0.6:
                hand_path_label = "Neutral (good for driver)"
            else:
                hand_path_label = "Shallow (ideal for driver)"
        else:
            # Iron can be steeper
            if steepness_ratio > 1.5:
                hand_path_label = "Steep (good for irons)"
            elif steepness_ratio > 0.8:
                hand_path_label = "Neutral (versatile)"
            else:
                hand_path_label = "Shallow (may need more compression)"

        df["hand_path_steepness"] = steepness_ratio
        df["hand_path_label"] = hand_path_label
        df["has_3d_data"] = has_3d

        print(f"DEBUG: Y change = {y_change:.4f}, Z/X change = {z_change:.4f}")
        print(f"DEBUG: Hand path steepness = {steepness_ratio:.2f}")
        print(f"DEBUG: Hand path = {hand_path_label}")

        # -----------------------------------
        # 6. Arm Extension at Impact
        # -----------------------------------
        def calculate_angle(p1, p2, p3):
            """Calculate angle at p2 formed by p1-p2-p3"""
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 < 1e-8 or norm2 < 1e-8:
                return 180.0
            
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            return angle

        # Check which columns are available
        has_shoulder = "shoulder_x_smooth" in df.columns
        has_elbow = "elbow_x_smooth" in df.columns
        has_wrist = "wrist_x_smooth" in df.columns

        if has_shoulder and has_elbow and has_wrist:
            if has_3d:
                shoulder_impact = [
                    df.loc[impact, "shoulder_x_smooth"],
                    df.loc[impact, "shoulder_y_smooth"],
                    df.loc[impact, "shoulder_z_smooth"]
                ]
                elbow_impact = [
                    df.loc[impact, "elbow_x_smooth"],
                    df.loc[impact, "elbow_y_smooth"],
                    df.loc[impact, "elbow_z_smooth"]
                ]
                wrist_impact = [
                    df.loc[impact, "wrist_x_smooth"],
                    df.loc[impact, "wrist_y_smooth"],
                    df.loc[impact, "wrist_z_smooth"]
                ]
            else:
                # 2D - use X, Y only (Z = 0)
                shoulder_impact = [
                    df.loc[impact, "shoulder_x_smooth"],
                    df.loc[impact, "shoulder_y_smooth"],
                    0
                ]
                elbow_impact = [
                    df.loc[impact, "elbow_x_smooth"],
                    df.loc[impact, "elbow_y_smooth"],
                    0
                ]
                wrist_impact = [
                    df.loc[impact, "wrist_x_smooth"],
                    df.loc[impact, "wrist_y_smooth"],
                    0
                ]
            
            # Elbow angle (180° = fully straight)
            elbow_angle = calculate_angle(shoulder_impact, elbow_impact, wrist_impact)
            
            # Adjusted thresholds for pose estimation data
            if elbow_angle > 155:
                arm_extension_label = "Excellent (fully extended)"
            elif elbow_angle > 140:
                arm_extension_label = "Good (slight bend is normal)"
            elif elbow_angle > 125:
                arm_extension_label = "Moderate"
            else:
                arm_extension_label = "Needs improvement (chicken wing)"
            
            df["elbow_angle_impact"] = elbow_angle
            df["arm_extension_label"] = arm_extension_label
            
            print(f"DEBUG: Elbow angle at impact = {elbow_angle:.1f}°")
            print(f"DEBUG: Arm extension = {arm_extension_label}")
        else:
            df["elbow_angle_impact"] = None
            df["arm_extension_label"] = "N/A (missing joint data)"
            elbow_angle = None

        # -----------------------------------
        # 7. Wrist Speed & Timing
        # -----------------------------------
        wrist_x = df["wrist_x_smooth"].values
        
        # Calculate velocity
        dx = np.diff(wrist_x)
        dy = np.diff(wrist_y)
        
        if has_3d:
            dz = np.diff(wrist_z)
            velocity = np.sqrt(dx**2 + dy**2 + dz**2)
        else:
            velocity = np.sqrt(dx**2 + dy**2)
        
        # Find max velocity during downswing
        downswing_start_idx = backswing_top
        downswing_end_idx = min(impact + 5, len(velocity))
        
        downswing_velocity = velocity[downswing_start_idx:downswing_end_idx]
        if len(downswing_velocity) > 0:
            max_velocity = np.max(downswing_velocity)
            max_velocity_frame = downswing_start_idx + np.argmax(downswing_velocity)
        else:
            max_velocity = 0
            max_velocity_frame = impact
        
        df["max_wrist_speed"] = max_velocity
        df["max_speed_frame"] = max_velocity_frame
        
        # Speed timing - max speed should be AT or JUST BEFORE impact
        frames_from_impact = max_velocity_frame - impact
        abs_frames_diff = abs(frames_from_impact)
        
        if abs_frames_diff <= 2:
            speed_timing = "Excellent (max speed at impact)"
            speed_timing_score = 100
        elif abs_frames_diff <= 4:
            speed_timing = "Good (max speed near impact)"
            speed_timing_score = 85
        elif abs_frames_diff <= 6:
            speed_timing = "Moderate"
            speed_timing_score = 70
        else:
            if frames_from_impact < 0:
                speed_timing = "Early release (max speed too early)"
            else:
                speed_timing = "Late release (max speed after impact)"
            speed_timing_score = 50
        
        df["speed_timing"] = speed_timing
        df["speed_timing_score"] = speed_timing_score
        df["max_speed_frames_from_impact"] = frames_from_impact
        
        print(f"DEBUG: Max wrist speed = {max_velocity:.4f} at frame {max_velocity_frame}")
        print(f"DEBUG: Frames from impact = {frames_from_impact}")
        print(f"DEBUG: Speed timing = {speed_timing}")

        # -----------------------------------
        # 8. Tempo calculation
        # -----------------------------------
        backswing_time = backswing_top - backswing_start
        downswing_time = impact - backswing_top
        tempo_ratio = round(backswing_time / downswing_time, 2) if downswing_time > 0 else 0.0

        df["tempo_ratio"] = tempo_ratio
        df["backswing_frames"] = backswing_time
        df["downswing_frames"] = downswing_time

        # -----------------------------------
        # 9. Finish detection
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

        # -----------------------------------
        # 10. Overall Swing Rating
        # -----------------------------------
        score = 0
        max_score = 0
        
        # Tempo (ideal: 2.0-3.5:1)
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
        
        # Hand path (different scoring for driver vs iron)
        max_score += 25
        if club_type.lower() == "driver":
            if 0.4 <= steepness_ratio <= 1.0:
                score += 25
            elif 0.3 <= steepness_ratio <= 1.2:
                score += 18
            else:
                score += 10
        else:
            if 0.8 <= steepness_ratio <= 1.5:
                score += 25
            elif 0.5 <= steepness_ratio <= 2.0:
                score += 18
            else:
                score += 10
        
        # Speed timing
        max_score += 25
        score += int(speed_timing_score * 0.25)
        
        overall_rating = round(score / max_score * 100) if max_score > 0 else 0
        
        if overall_rating >= 85:
            rating_label = "Excellent"
        elif overall_rating >= 70:
            rating_label = "Good"
        elif overall_rating >= 55:
            rating_label = "Average"
        else:
            rating_label = "Needs Work"
        
        df["overall_score"] = overall_rating
        df["overall_rating"] = rating_label

        return df
        
    except Exception as e:
        df["error"] = str(e)
        print(f"Swing analysis failed: {e}")
        return df


def print_analysis_results(df):
    """Print formatted analysis results"""
    print("\n" + "="*60)
    print("🏌️ SWING ANALYSIS RESULTS")
    print("="*60)
    
    # Club type and data type
    club_type = df['club_type'].iloc[0] if 'club_type' in df.columns else "iron"
    has_3d = df['has_3d_data'].iloc[0] if 'has_3d_data' in df.columns else False
    print(f"\n🏌️ Club Type: {club_type.upper()}")
    print(f"📊 Data Type: {'3D (CoMotion/SMPL)' if has_3d else '2D (MediaPipe)'}")
    
    print(f"\n📍 PHASE DETECTION:")
    print(f"   Backswing Start:         Frame {int(df['backswing_start_idx'].iloc[0])}")
    print(f"   Top of Backswing:        Frame {int(df['backswing_top_idx'].iloc[0])}")
    print(f"   Impact:                  Frame {int(df['impact_idx'].iloc[0])}")
    print(f"   Finish:                  Frame {int(df['finish_idx'].iloc[0])}")
    
    print(f"\n⏱️ TEMPO:")
    print(f"   Backswing:      {int(df['backswing_frames'].iloc[0])} frames")
    print(f"   Downswing:      {int(df['downswing_frames'].iloc[0])} frames")
    print(f"   Follow-through: {int(df['follow_through_frames'].iloc[0])} frames")
    print(f"   Tempo Ratio:    {df['tempo_ratio'].iloc[0]}:1")
    
    ratio = df['tempo_ratio'].iloc[0]
    if 2.0 <= ratio <= 3.5:
        print(f"   ✅ Good tempo (tour average is ~3:1)")
    elif ratio < 2.0:
        print(f"   ⚠️ Quick backswing - try slowing down")
    else:
        print(f"   ⚠️ Slow downswing - try accelerating through impact")
    
    print(f"\n🖐️ HAND PATH:")
    print(f"   Steepness Ratio: {df['hand_path_steepness'].iloc[0]:.2f}")
    print(f"   Classification:  {df['hand_path_label'].iloc[0]}")
    
    print(f"\n💪 ARM EXTENSION AT IMPACT:")
    if df['elbow_angle_impact'].iloc[0] is not None:
        print(f"   Elbow Angle:    {df['elbow_angle_impact'].iloc[0]:.1f}°")
        print(f"   Classification: {df['arm_extension_label'].iloc[0]}")
    else:
        print(f"   Data not available")
    
    print(f"\n⚡ SPEED TIMING:")
    print(f"   Max Speed Frame:    {int(df['max_speed_frame'].iloc[0])}")
    print(f"   Impact Frame:       {int(df['impact_idx'].iloc[0])}")
    print(f"   Frames Difference:  {int(df['max_speed_frames_from_impact'].iloc[0])}")
    print(f"   Assessment:         {df['speed_timing'].iloc[0]}")
    
    print(f"\n🏆 OVERALL RATING:")
    print(f"   Score:  {df['overall_score'].iloc[0]}/100")
    print(f"   Rating: {df['overall_rating'].iloc[0]}")
    
    print("="*60)
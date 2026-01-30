import matplotlib.pyplot as plt
import numpy as np


def plot_wrist_timeline(df):
    """Plot wrist Y movement over time with phase markers"""
    fig, ax = plt.subplots(figsize=(10, 4))

    y = df["wrist_y_smooth"]
    ax.plot(y, label="Wrist Height (Y)", linewidth=2)

    # Phase markers
    bs = int(df["backswing_start_idx"].iloc[0])
    top = int(df["backswing_top_idx"].iloc[0])
    impact = int(df["impact_idx"].iloc[0])
    finish = int(df["finish_idx"].iloc[0])

    ax.axvline(bs, color="green", linestyle="--", label="Backswing Start")
    ax.axvline(top, color="orange", linestyle="--", label="Top of Backswing")
    ax.axvline(impact, color="red", linestyle="--", label="Impact")
    ax.axvline(finish, color="purple", linestyle="--", label="Finish")

    ax.set_title("Wrist Height Timeline")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Wrist Y Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_hand_path(df):
    """Plot hand path - bird's eye view or 2D trajectory"""
    fig, ax = plt.subplots(figsize=(6, 6))

    has_3d = df['has_3d_data'].iloc[0] if 'has_3d_data' in df.columns else False
    
    if has_3d and "wrist_z_smooth" in df.columns:
        x = df["wrist_z_smooth"]  # Forward/back
        y = df["wrist_x_smooth"]  # Side to side
        ax.set_xlabel("Forward/Back (Z)")
        ax.set_ylabel("Side to Side (X)")
        ax.set_title("Bird's Eye View — Hand Path")
    else:
        x = df["wrist_x_smooth"]
        y = df["wrist_y_smooth"]
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Hand Path (2D)")
        ax.invert_yaxis()  # Invert for screen coordinates

    # Plot full path
    ax.plot(x, y, 'b-', linewidth=1, alpha=0.5, label='Full swing')
    
    # Highlight downswing
    top = int(df["backswing_top_idx"].iloc[0])
    impact = int(df["impact_idx"].iloc[0])
    ax.plot(x[top:impact+1], y[top:impact+1], 'r-', linewidth=2, label='Downswing')
    
    # Mark key points
    bs = int(df["backswing_start_idx"].iloc[0])
    ax.scatter(x[bs], y[bs], color='green', s=100, zorder=5, label='Address')
    ax.scatter(x[top], y[top], color='orange', s=100, zorder=5, label='Top')
    ax.scatter(x[impact], y[impact], color='red', s=100, zorder=5, label='Impact')

    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    return fig


def plot_tempo(df):
    """Visual display of tempo ratio"""
    bs = int(df["backswing_start_idx"].iloc[0])
    top = int(df["backswing_top_idx"].iloc[0])
    impact = int(df["impact_idx"].iloc[0])
    finish = int(df["finish_idx"].iloc[0])

    backswing_time = top - bs
    downswing_time = impact - top
    follow_through_time = finish - impact

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(
        ["Backswing", "Downswing", "Follow-through"], 
        [backswing_time, downswing_time, follow_through_time],
        color=['#3498db', '#e74c3c', '#9b59b6']
    )

    # Add value labels on bars
    for bar, val in zip(bars, [backswing_time, downswing_time, follow_through_time]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val} frames', ha='center', va='bottom', fontsize=10)

    ratio = round(backswing_time / downswing_time, 2) if downswing_time > 0 else 0
    ax.set_title(f"Swing Tempo — {ratio}:1 (Backswing : Downswing)")
    ax.set_ylabel("Frames")

    return fig


def plot_speed_profile(df):
    """Plot wrist speed throughout the swing"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Calculate velocity
    wrist_x = df["wrist_x_smooth"].values
    wrist_y = df["wrist_y_smooth"].values
    
    dx = np.diff(wrist_x)
    dy = np.diff(wrist_y)
    
    if "wrist_z_smooth" in df.columns:
        wrist_z = df["wrist_z_smooth"].values
        dz = np.diff(wrist_z)
        velocity = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        velocity = np.sqrt(dx**2 + dy**2)
    
    ax.plot(velocity, 'purple', linewidth=2, label='Wrist Speed')
    
    # Phase markers
    top = int(df["backswing_top_idx"].iloc[0])
    impact = int(df["impact_idx"].iloc[0])
    max_speed_frame = int(df["max_speed_frame"].iloc[0])
    
    ax.axvline(top, color="orange", linestyle="--", label="Top of Backswing")
    ax.axvline(impact, color="red", linestyle="--", label="Impact")
    ax.axvline(max_speed_frame, color="green", linestyle=":", linewidth=2, label="Max Speed")
    
    ax.set_title("Wrist Speed Profile")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Speed (units/frame)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_overall_score(df):
    """Visual display of overall score breakdown"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Get scores
    tempo_ratio = df['tempo_ratio'].iloc[0]
    steepness = df['hand_path_steepness'].iloc[0]
    elbow_angle = df['elbow_angle_impact'].iloc[0]
    speed_score = df['speed_timing_score'].iloc[0]
    
    # Calculate individual scores (same logic as in analyze_swing)
    if 2.0 <= tempo_ratio <= 3.5:
        tempo_score = 100
    elif 1.5 <= tempo_ratio <= 4.0:
        tempo_score = 72
    else:
        tempo_score = 40
    
    if 0.8 <= steepness <= 1.5:
        path_score = 100
    elif 0.5 <= steepness <= 2.0:
        path_score = 72
    else:
        path_score = 40
    
    if elbow_angle is not None:
        if elbow_angle > 155:
            arm_score = 100
        elif elbow_angle > 140:
            arm_score = 88
        elif elbow_angle > 125:
            arm_score = 60
        else:
            arm_score = 32
    else:
        arm_score = 50  # Default if no data
    
    categories = ['Tempo', 'Hand Path', 'Arm Extension', 'Speed Timing']
    scores = [tempo_score, path_score, arm_score, speed_score]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    bars = ax.barh(categories, scores, color=colors)
    
    # Add score labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                f'{score}%', ha='left', va='center', fontsize=10)
    
    ax.set_xlim(0, 110)
    ax.set_xlabel("Score (%)")
    ax.set_title(f"Swing Analysis Breakdown\nOverall: {df['overall_score'].iloc[0]}/100 ({df['overall_rating'].iloc[0]})")
    ax.axvline(70, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
    
    return fig
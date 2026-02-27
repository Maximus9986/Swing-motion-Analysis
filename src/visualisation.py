import matplotlib.pyplot as plt
import numpy as np


def _safe_get(df, col, default=np.nan):
    return df[col].iloc[0] if col in df.columns and len(df) > 0 else default


def plot_wrist_timeline(df):
    """Plot wrist Y movement over time with phase markers"""
    fig, ax = plt.subplots(figsize=(10, 4))

    if "wrist_y_smooth" not in df.columns:
        ax.set_title("Wrist Height Timeline (missing wrist_y_smooth)")
        return fig

    y = df["wrist_y_smooth"].values
    ax.plot(y, label="Wrist Height (Y)", linewidth=2)

    # Safe markers
    bs = int(_safe_get(df, "backswing_start_idx", 0))
    top = int(_safe_get(df, "backswing_top_idx", 0))
    impact = int(_safe_get(df, "impact_idx", 0))
    finish = int(_safe_get(df, "finish_idx", len(df) - 1))

    ax.plot(y, color="#2980b9", linewidth=2, label="Wrist Height (Y)")

    ax.axvline(bs, color="#27ae60", linestyle="--", label="Backswing Start")
    ax.axvline(top, color="#f39c12", linestyle="--", label="Top of Backswing")
    ax.axvline(impact, color="#c0392b", linestyle="--", label="Impact")
    ax.axvline(finish, color="#8e44ad", linestyle="--", label="Finish")


    ax.set_title("Wrist Height Timeline")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Wrist Y Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_hand_path(df):
    """Purely visual hand path (NOT used in scoring)."""
    fig, ax = plt.subplots(figsize=(6, 6))

    if "wrist_x_smooth" not in df.columns or "wrist_y_smooth" not in df.columns:
        ax.set_title("Hand Path (missing wrist data)")
        return fig

    has_3d = bool(df["has_3d_data"].iloc[0]) if "has_3d_data" in df.columns else False

    if has_3d and "wrist_z_smooth" in df.columns:
        x = df["wrist_z_smooth"].values
        y = df["wrist_x_smooth"].values
        ax.set_xlabel("Forward / Back (Z)")
        ax.set_ylabel("Side to Side (X)")
        ax.set_title("Hand Path — Bird's Eye View")
    else:
        x = df["wrist_x_smooth"].values
        y = df["wrist_y_smooth"].values
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Hand Path (2D)")
        ax.invert_yaxis()

    # --- Colours restored ---
    ax.plot(x, y, color="#3498db", linewidth=1.5, alpha=0.6, label="Full swing")

    top = int(df["backswing_top_idx"].iloc[0])
    impact = int(df["impact_idx"].iloc[0])

    if impact > top:
        ax.plot(
            x[top:impact + 1],
            y[top:impact + 1],
            color="#e74c3c",
            linewidth=2.5,
            label="Downswing"
        )

    bs = int(df["backswing_start_idx"].iloc[0])

    ax.scatter(x[bs], y[bs], color="#2ecc71", s=90, zorder=5, label="Address")
    ax.scatter(x[top], y[top], color="#f39c12", s=90, zorder=5, label="Top")
    ax.scatter(x[impact], y[impact], color="#c0392b", s=90, zorder=5, label="Impact")

    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    return fig


def plot_tempo(df):
    """Visual display of tempo ratio"""
    bs = int(_safe_get(df, "backswing_start_idx", 0))
    top = int(_safe_get(df, "backswing_top_idx", 0))
    impact = int(_safe_get(df, "impact_idx", 0))
    finish = int(_safe_get(df, "finish_idx", len(df) - 1))

    backswing_time = max(0, top - bs)
    downswing_time = max(1, impact - top)     # avoid /0
    follow_through_time = max(0, finish - impact)

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(
        ["Backswing", "Downswing", "Follow-through"],
        [backswing_time, downswing_time, follow_through_time]
    )

    for bar, val in zip(bars, [backswing_time, downswing_time, follow_through_time]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val} frames", ha="center", va="bottom", fontsize=10)

    ratio = round(backswing_time / downswing_time, 2) if downswing_time > 0 else 0
    ax.set_title(f"Swing Tempo — {ratio}:1 (Backswing : Downswing)")
    ax.set_ylabel("Frames")

    return fig


def plot_speed_profile(df):
    """Plot wrist speed throughout the swing"""
    fig, ax = plt.subplots(figsize=(10, 4))

    if "wrist_x_smooth" not in df.columns or "wrist_y_smooth" not in df.columns:
        ax.set_title("Wrist Speed Profile (missing wrist_x_smooth / wrist_y_smooth)")
        return fig

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

    ax.plot(velocity, linewidth=2, label="Wrist Speed")

    top = int(_safe_get(df, "backswing_top_idx", 0))
    impact = int(_safe_get(df, "impact_idx", 0))
    max_speed_frame = int(_safe_get(df, "max_speed_frame", impact))

    ax.axvline(top, linestyle="--", label="Top of Backswing")
    ax.axvline(impact, linestyle="--", label="Impact")
    ax.axvline(max_speed_frame, linestyle=":", linewidth=2, label="Max Speed")

    ax.set_title("Wrist Speed Profile")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Speed (units/frame)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_overall_score(df):
    """
    Score breakdown WITHOUT hand_path_steepness.
    Uses:
      - Tempo
      - Arm extension
      - Speed timing
      - Early Extension (if available)
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    tempo_ratio = _safe_get(df, "tempo_ratio", np.nan)
    elbow_angle = _safe_get(df, "elbow_angle_impact", np.nan)
    speed_score = _safe_get(df, "speed_timing_score", 0)
    ee_score = _safe_get(df, "early_extension_score", np.nan)  # optional

    # Tempo -> 0..100
    if np.isfinite(tempo_ratio):
        tempo_ratio = float(tempo_ratio)
        if 2.0 <= tempo_ratio <= 3.5:
            tempo_score = 100
        elif 1.5 <= tempo_ratio <= 4.0:
            tempo_score = 72
        else:
            tempo_score = 40
    else:
        tempo_score = 50

    # Arm extension -> 0..100
    if np.isfinite(elbow_angle):
        elbow_angle = float(elbow_angle)
        if elbow_angle > 155:
            arm_score = 100
        elif elbow_angle > 140:
            arm_score = 88
        elif elbow_angle > 125:
            arm_score = 60
        else:
            arm_score = 32
    else:
        arm_score = 50

    # Speed timing already 0..100
    speed_score = int(speed_score) if np.isfinite(speed_score) else 0

    categories = ["Tempo", "Arm Extension", "Speed Timing"]
    scores = [tempo_score, arm_score, speed_score]

    # Optional Early Extension
    if np.isfinite(ee_score):
        # If your early_extension_score is "badness" 0..100, convert to "goodness"
        ee_good = int(max(0, min(100, 100 - float(ee_score))))
        categories.append("Early Extension")
        scores.append(ee_good)

    colors = []
    for s in scores:
        if s >= 80:
            colors.append("#2ecc71")   # green
        elif s >= 60:
            colors.append("#f1c40f")   # yellow
        else:
            colors.append("#e74c3c")   # red

    bars = ax.barh(categories, scores, color=colors)


    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{int(s)}%", ha="left", va="center", fontsize=10)

    overall = _safe_get(df, "overall_score", 0)
    rating = _safe_get(df, "overall_rating", "N/A")

    ax.set_xlim(0, 110)
    ax.set_xlabel("Score (%)")
    ax.set_title(f"Swing Analysis Breakdown\nOverall: {overall}/100 ({rating})")
    ax.axvline(70, linestyle="--", alpha=0.5)

    return fig

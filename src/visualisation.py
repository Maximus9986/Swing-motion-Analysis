import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------
# PLOT 1 — Wrist Height Timeline (Y axis)
# ---------------------------------------------------
def plot_wrist_timeline(df):
    """
    Plot wrist Y movement over time + key swing phases:
        • Backswing start
        • Top of backswing
        • Impact
    """

    fig, ax = plt.subplots(figsize=(10, 4))

    y = df["wrist_y_smooth"]

    ax.plot(y, label="Wrist Height (Y)", linewidth=2)

    # Phase markers
    bs = int(df["backswing_start_idx"].iloc[0])
    top = int(df["backswing_top_idx"].iloc[0])
    impact = int(df["impact_idx"].iloc[0])

    ax.axvline(bs, color="green", linestyle="--", label="Backswing Start")
    ax.axvline(top, color="orange", linestyle="--", label="Top of Backswing")
    ax.axvline(impact, color="red", linestyle="--", label="Impact")

    ax.set_title("Wrist Height Timeline")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Wrist Y Height")
    ax.legend()

    return fig


# ---------------------------------------------------
# PLOT 2 — Bird's-eye Swing Path (X vs Z)
# ---------------------------------------------------
def plot_birds_eye(df):
    """
    Shows the hand path from a top-down view (X vs Z)
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    x = df["wrist_x_smooth"]
    z = df["wrist_z_smooth"]

    ax.plot(z, x, linewidth=2)

    ax.set_title("Bird’s Eye View — Swing Path")
    ax.set_xlabel("Forward/Back (Z)")
    ax.set_ylabel("Side to Side (X)")

    ax.grid(True)

    return fig


# ---------------------------------------------------
# PLOT 3 — Tempo Ratio Visualization
# ---------------------------------------------------
def plot_tempo(df):
    """
    Visual display of tempo ratio:
        backswing_time : downswing_time
    """

    bs = int(df["backswing_start_idx"].iloc[0])
    top = int(df["backswing_top_idx"].iloc[0])
    impact = int(df["impact_idx"].iloc[0])

    backswing_time = top - bs
    downswing_time = impact - top

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(["Backswing", "Downswing"], [backswing_time, downswing_time])

    ratio = round(backswing_time / downswing_time, 2) if downswing_time > 0 else 0
    ax.set_title(f"Swing Tempo — {ratio}:1 (Backswing : Downswing)")

    return fig

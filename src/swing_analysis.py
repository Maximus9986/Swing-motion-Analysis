import numpy as np
import pandas as pd

def analyze_swing(df):
    """
    Smooth the wrist 3D coordinates and calculate total wrist motion.
    Works with columns: x, y, z
    """
    df["wrist_x_smooth"] = df["x"].rolling(window=5, min_periods=1).mean()
    df["wrist_y_smooth"] = df["y"].rolling(window=5, min_periods=1).mean()
    df["wrist_z_smooth"] = df["z"].rolling(window=5, min_periods=1).mean()

    # 3D wrist movement magnitude
    df["wrist_motion"] = np.sqrt(
        df["wrist_x_smooth"]**2 +
        df["wrist_y_smooth"]**2 +
        df["wrist_z_smooth"]**2
    )
    return df

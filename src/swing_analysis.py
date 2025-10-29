import pandas as pd

def analyze_swing(frame_data):
    """
    Converts frame_data into a pandas DataFrame for plotting or analysis.
    """
    df = pd.DataFrame(frame_data)
    # Optional: smooth wrist_y with rolling average
    df["wrist_y_smooth"] = df["wrist_y"].rolling(window=5, min_periods=1).mean()
    return df

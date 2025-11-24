"""
Configuration file for Golf Swing Analysis
Contains all constants and thresholds
"""

# === DATA SMOOTHING ===
SMOOTHING_WINDOW = 5  # frames for rolling average
MIN_PERIODS = 1

# === SWING PHASE DETECTION ===
MIN_VIDEO_FRAMES = 30  # minimum frames required for analysis
BACKSWING_SEARCH_START = 5  # frames to skip at start
IMPACT_SEARCH_WINDOW = 5  # frames after downswing to search for impact

# === SWING PATH CLASSIFICATION ===
# Angles in degrees for swing path classification
SWING_PATH_THRESHOLDS = {
    "strong_in_to_out": 8,      # > 8° = Strong In-to-Out (Hook)
    "moderate_in_to_out": 3,    # 3-8° = Moderate In-to-Out (Draw)
    "neutral_upper": 3,         # -3 to 3° = Neutral/Straight
    "neutral_lower": -3,
    "moderate_out_to_in": -8,   # -8 to -3° = Moderate Out-to-In (Fade)
}

# === SWING TEMPO ===
IDEAL_TEMPO_RATIO = 3.0  # Ideal backswing:downswing ratio
TEMPO_TOLERANCE = 0.5    # Acceptable deviation

# === ANALYSIS PARAMETERS ===
SWING_PATH_SAMPLE_FRAMES = 10  # Number of frames for swing path regression
MIN_R_SQUARED = 0.5  # Minimum R² for valid path regression

# === FILE PATHS ===
TEMP_VIDEO_PATH = "temp.mp4"
OVERLAY_VIDEO_PATH = "overlay.mp4"
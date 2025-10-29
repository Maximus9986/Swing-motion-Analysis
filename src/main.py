from pose_tracking import extract_pose
from swing_analysis import analyze_swing
from visualisation import plot_swing
import pandas as pd

video_path = "../data/sample_swing1.mp4"
output_video = "../data/sample_swing1_overlay.mp4"

# Pose extraction + overlay video
frame_data = extract_pose(video_path, output_path=output_video)
df = pd.DataFrame(frame_data)

# Swing analysis
df = analyze_swing(df) 

# Plot results
plot_swing(df["wrist_y_smooth"])  

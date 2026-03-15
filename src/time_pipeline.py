import time
import cv2
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, r'C:\Users\User\FYP\Swing-motion-Analysis\src')
from pose_tracking import extract_pose
from swing_analysis import analyze_swing
from club_tracking import extract_clubhead_y_from_yolo

VIDEO_DIR = r'C:\Users\User\FYP\Swing-motion-Analysis\Data\video_160'
YOLO      = r'C:\Users\User\FYP\Swing-motion-Analysis\src\models\best.pt'
VIDEOS    = ['1119.mp4', '1152.mp4', '1202.mp4', '1286.mp4', '1378.mp4']

results = []

for filename in VIDEOS:
    video_path = f'{VIDEO_DIR}\\{filename}'

    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total_frames / fps

    print(f'\n{filename} ({duration:.1f}s, {total_frames} frames @ {fps:.0f}fps)')

    # Pose
    t0 = time.time()
    pose_rows = extract_pose(video_path, output_path=None, write_overlay=False, use_3d=False)
    t_pose = time.time() - t0

    # YOLO
    t0 = time.time()
    club_df = extract_clubhead_y_from_yolo(video_path, YOLO, conf=0.25, clubhead_kpt_idx=0)
    t_yolo = time.time() - t0

    # Analysis
    pose_df = pd.DataFrame(pose_rows)
    pose_df['frame'] = np.arange(len(pose_df))
    pose_df = pose_df.merge(
        club_df[['frame','clubhead_y','clubhead_valid','clubhead_y_smooth']],
        on='frame', how='left'
    )
    t0 = time.time()
    analyze_swing(pose_df, fps=fps, debug=False)
    t_analysis = time.time() - t0

    t_total = t_pose + t_yolo + t_analysis
    print(f'  Pose     : {t_pose:.2f}s ({t_pose/t_total*100:.1f}%)')
    print(f'  YOLO     : {t_yolo:.2f}s ({t_yolo/t_total*100:.1f}%)')
    print(f'  Analysis : {t_analysis:.2f}s ({t_analysis/t_total*100:.1f}%)')
    print(f'  Total    : {t_total:.2f}s')

    results.append({
        'file': filename, 'duration': round(duration, 1),
        't_pose': t_pose, 't_yolo': t_yolo,
        't_analysis': t_analysis, 't_total': t_total
    })

# Summary
df = pd.DataFrame(results)
print('\n' + '='*50)
print('MEAN ACROSS ALL VIDEOS')
print('='*50)
print(f'  Mean total time : {df["t_total"].mean():.2f}s  (std: {df["t_total"].std():.2f}s)')
print(f'  Mean pose time  : {df["t_pose"].mean():.2f}s  ({df["t_pose"].mean()/df["t_total"].mean()*100:.1f}%)')
print(f'  Mean YOLO time  : {df["t_yolo"].mean():.2f}s  ({df["t_yolo"].mean()/df["t_total"].mean()*100:.1f}%)')
print(f'  Mean analysis   : {df["t_analysis"].mean():.2f}s  ({df["t_analysis"].mean()/df["t_total"].mean()*100:.1f}%)')
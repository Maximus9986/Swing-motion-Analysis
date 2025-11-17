# src/pose_tracking.py
import cv2
import mediapipe as mp
import pandas as pd

def extract_pose(video_path, output_path=None, write_overlay=False):
    """
    Extract wrist pose from video using MediaPipe Pose.
    Optionally writes an overlay video showing detected landmarks.
    
    Returns:
        List of dicts with frame, wrist_x, wrist_y, wrist_z
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer if overlay requested
    if write_overlay and output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pose_data = []
    frame_idx = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            wrist_x, wrist_y, wrist_z = None, None, None

            if results.pose_landmarks:
                landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                wrist_x = landmark.x
                wrist_y = landmark.y
                wrist_z = landmark.z

                if write_overlay:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            pose_data.append({
                "frame": frame_idx,
                "wrist_x": wrist_x,
                "wrist_y": wrist_y,
                "wrist_z": wrist_z
            })

            if write_overlay and output_path:
                out.write(frame)

            frame_idx += 1

    cap.release()
    if write_overlay and output_path:
        out.release()

    # Drop frames where wrist is not detected
    df = pd.DataFrame(pose_data)
    df = df.dropna(subset=["wrist_x", "wrist_y", "wrist_z"])

    return df.to_dict(orient="records")

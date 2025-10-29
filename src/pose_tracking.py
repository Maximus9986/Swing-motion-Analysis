# pose_tracking.py
import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def extract_pose(video_path, output_path=None):
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    frame_data = []  # list of dicts: {"frame_idx":..., "wrist_y":...}

    writer = None
    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        wrist_y = None
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # Choose right wrist for example
            wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            wrist_y = wrist.y * int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        frame_data.append({"frame_idx": frame_idx, "wrist_y": wrist_y})

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
        print(f"✅ Overlay video saved to: {output_path}")

    pose.close()
    cv2.destroyAllWindows()
    return frame_data


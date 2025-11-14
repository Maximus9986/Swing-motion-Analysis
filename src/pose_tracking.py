import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def extract_pose(video_path, output_path="overlay.mp4"):
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    frame_data = []

    # ---- Read video info safely ----
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30   # fallback to safe FPS

    print(f"[INFO] Width={width}, Height={height}, FPS={fps}")

    # ---- Create writer ----
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("[ERROR] Could not open writer for overlay.mp4")

    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        wrist_x = wrist_y = wrist_z = None

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Save frame data
        frame_data.append({
            "frame": frame_idx,
            "x": wrist_x,
            "y": wrist_y,
            "z": wrist_z
        })

        # Write overlay frame
        writer.write(frame)

    cap.release()
    writer.release()
    pose.close()

    # Final check
    try:
        import os
        size = os.path.getsize(output_path)
        print(f"[INFO] overlay.mp4 created ({size} bytes)")
        if size == 0:
            print("[ERROR] overlay.mp4 is empty!")
    except:
        print("[ERROR] overlay.mp4 was not created!")

    return frame_data

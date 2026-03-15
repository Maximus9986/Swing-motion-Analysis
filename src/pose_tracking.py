import cv2
import mediapipe as mp
import pandas as pd
import numpy as np


def extract_pose_2d(video_path, output_path=None, write_overlay=False, handed="right"):
    """
    Extract 2D pose from video using MediaPipe Pose (world landmarks).
    Tracks the LEAD-ARM side joints (wrist, elbow, shoulder, hip).

    For a right-handed golfer the lead arm is the LEFT side.
    For a left-handed golfer the lead arm is the RIGHT side.

    Parameters
    ----------
    video_path : str
        Path to input video file.
    output_path : str, optional
        Path to save overlay video.
    write_overlay : bool
        Whether to create overlay video with pose landmarks.
    handed : str
        "right" (default) or "left".  Selects which side joints to track.

    Returns
    -------
    list of dict
        Pose data for each frame with all joint coordinates.
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

    print(f"[INFO] Video: {width}x{height} @ {fps:.1f}fps")
    print(f"[INFO] Handed: {handed} (lead arm = {'LEFT' if handed == 'right' else 'RIGHT'})")

    # Select lead-arm landmarks based on handedness
    if handed.lower() == "left":
        WRIST  = mp_pose.PoseLandmark.RIGHT_WRIST
        ELBOW  = mp_pose.PoseLandmark.RIGHT_ELBOW
        SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
        HIP    = mp_pose.PoseLandmark.RIGHT_HIP
    else:
        WRIST  = mp_pose.PoseLandmark.LEFT_WRIST
        ELBOW  = mp_pose.PoseLandmark.LEFT_ELBOW
        SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
        HIP    = mp_pose.PoseLandmark.LEFT_HIP

    # Setup video writer if overlay requested
    out = None
    if write_overlay and output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pose_data = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            frame_data = {"frame": frame_idx}

            if results.pose_world_landmarks:
                landmarks = results.pose_world_landmarks.landmark

                lm_wrist   = landmarks[WRIST]
                lm_elbow   = landmarks[ELBOW]
                lm_shoulder = landmarks[SHOULDER]
                lm_hip     = landmarks[HIP]

                frame_data.update({
                    "wrist_x": lm_wrist.x,
                    "wrist_y": lm_wrist.y,
                    "wrist_z": lm_wrist.z,
                    "wrist_visibility": lm_wrist.visibility,

                    "elbow_x": lm_elbow.x,
                    "elbow_y": lm_elbow.y,
                    "elbow_z": lm_elbow.z,
                    "elbow_visibility": lm_elbow.visibility,

                    "shoulder_x": lm_shoulder.x,
                    "shoulder_y": lm_shoulder.y,
                    "shoulder_z": lm_shoulder.z,
                    "shoulder_visibility": lm_shoulder.visibility,

                    "hip_x": lm_hip.x,
                    "hip_y": lm_hip.y,
                    "hip_z": lm_hip.z,
                    "hip_visibility": lm_hip.visibility,
                })

                if write_overlay:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )
            else:
                for joint in ["wrist", "elbow", "shoulder", "hip"]:
                    for axis in ["x", "y", "z", "visibility"]:
                        frame_data[f"{joint}_{axis}"] = None

            pose_data.append(frame_data)

            if write_overlay and out is not None:
                out.write(frame)

            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"[INFO] Processed {frame_idx} frames...")

    cap.release()
    if out is not None:
        out.release()

    print(f"[INFO] Completed: {frame_idx} frames processed")

    df = pd.DataFrame(pose_data)

    numeric_cols = [col for col in df.columns if col not in ['frame']]
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

    if 'wrist_visibility' in df.columns:
        detection_rate = df['wrist_visibility'].notna().sum() / len(df) * 100
        print(f"[INFO] Detection rate: {detection_rate:.1f}%")

    return df.to_dict(orient="records")


def extract_pose_3d(video_path):
    """
    Placeholder for 3D pose extraction with HybrIK.
    This will be implemented on Kaggle with GPU.
    """
    raise NotImplementedError(
        "3D pose extraction with HybrIK requires GPU.\n"
        "Please run the Kaggle notebook: kaggle_hybrik_notebook.ipynb\n"
        "Then upload the resulting CSV to continue."
    )


def extract_pose(video_path, output_path=None, write_overlay=False, use_3d=False, handed="right"):
    """
    Unified pose extraction interface.

    Parameters
    ----------
    video_path : str
        Path to input video file.
    output_path : str, optional
        Path to save overlay video.
    write_overlay : bool
        Whether to create overlay video with pose landmarks.
    use_3d : bool
        If True, attempts HybrIK 3D (requires GPU/Kaggle).
        If False or fails, uses MediaPipe world landmarks.
    handed : str
        "right" (default) or "left".  Passed through to extract_pose_2d.

    Returns
    -------
    list of dict
        Pose data for each frame.
    """
    if use_3d:
        try:
            print("[INFO] Attempting HybrIK 3D pose estimation...")
            return extract_pose_3d(video_path)
        except NotImplementedError as e:
            print(f"[WARN] {e}")
            print("[INFO] Falling back to MediaPipe 3D world landmarks")
        except Exception as e:
            print(f"[ERROR] HybrIK failed: {e}")
            print("[INFO] Falling back to MediaPipe 3D world landmarks")

    print("[INFO] Using MediaPipe Pose (3D world landmarks)")
    return extract_pose_2d(
        video_path,
        output_path=output_path,
        write_overlay=write_overlay,
        handed=handed
    )
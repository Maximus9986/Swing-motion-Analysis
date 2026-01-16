import cv2
import mediapipe as mp
import pandas as pd
import numpy as np


def extract_pose_2d(video_path, output_path=None, write_overlay=False):
    """
    Extract 3D pose from video using MediaPipe Pose.
    Uses pose_world_landmarks for TRUE 3D coordinates in meters.
    Tracks wrist, elbow, shoulder, and hip for full swing analysis.
    
    Parameters:
    -----------
    video_path : str
        Path to input video file
    output_path : str, optional
        Path to save overlay video
    write_overlay : bool
        Whether to create overlay video with pose landmarks
    
    Returns:
    --------
    list of dict
        Pose data for each frame with all joint coordinates
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

    # Setup video writer if overlay requested
    out = None
    if write_overlay and output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pose_data = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # 0=Lite, 1=Full, 2=Heavy (most accurate)
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # Initialize frame data
            frame_data = {"frame": frame_idx}

            # Extract 3D WORLD coordinates (TRUE 3D in meters)
            if results.pose_world_landmarks:
                landmarks = results.pose_world_landmarks.landmark
                
                # LEFT side joints (lead arm for right-handed golfer)
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                
                # Store coordinates
                frame_data.update({
                    "wrist_x": left_wrist.x,
                    "wrist_y": left_wrist.y,
                    "wrist_z": left_wrist.z,
                    "wrist_visibility": left_wrist.visibility,
                    
                    "elbow_x": left_elbow.x,
                    "elbow_y": left_elbow.y,
                    "elbow_z": left_elbow.z,
                    "elbow_visibility": left_elbow.visibility,
                    
                    "shoulder_x": left_shoulder.x,
                    "shoulder_y": left_shoulder.y,
                    "shoulder_z": left_shoulder.z,
                    "shoulder_visibility": left_shoulder.visibility,
                    
                    "hip_x": left_hip.x,
                    "hip_y": left_hip.y,
                    "hip_z": left_hip.z,
                    "hip_visibility": left_hip.visibility,
                })
                
                # Draw overlay if requested
                if write_overlay:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )
            else:
                # No pose detected - fill with None
                for joint in ["wrist", "elbow", "shoulder", "hip"]:
                    for axis in ["x", "y", "z", "visibility"]:
                        frame_data[f"{joint}_{axis}"] = None

            pose_data.append(frame_data)

            # Write overlay frame
            if write_overlay and out is not None:
                out.write(frame)

            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 30 == 0:
                print(f"[INFO] Processed {frame_idx} frames...")

    cap.release()
    if out is not None:
        out.release()

    print(f"[INFO] Completed: {frame_idx} frames processed")
    
    # Convert to DataFrame
    df = pd.DataFrame(pose_data)
    
    # DON'T dropna() - keep all frames!
    # Instead, interpolate missing values for better continuity
    numeric_cols = [col for col in df.columns if col not in ['frame']]
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    
    # Report detection quality
    if 'wrist_visibility' in df.columns:
        detection_rate = df['wrist_visibility'].notna().sum() / len(df) * 100
        print(f"[INFO] Detection rate: {detection_rate:.1f}%")
    
    return df.to_dict(orient="records")


def extract_pose_3d(video_path):
    """
    Placeholder for 3D pose extraction with HybrIK.
    This will be implemented on Kaggle with GPU.
    
    TODO: Implement HybrIK processing on Kaggle
    Returns same format as extract_pose_2d for compatibility
    """
    raise NotImplementedError(
        "3D pose extraction with HybrIK requires GPU.\n"
        "Please run the Kaggle notebook: kaggle_hybrik_notebook.ipynb\n"
        "Then upload the resulting CSV to continue."
    )


def extract_pose(video_path, output_path=None, write_overlay=False, use_3d=False):
    """
    Unified pose extraction interface.
    
    Parameters:
    -----------
    video_path : str
        Path to input video file
    output_path : str, optional
        Path to save overlay video
    write_overlay : bool
        Whether to create overlay video with pose landmarks
    use_3d : bool
        If True, attempts HybrIK 3D (requires GPU/Kaggle)
        If False or fails, uses MediaPipe 3D world landmarks
    
    Returns:
    --------
    list of dict
        Pose data for each frame
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
        write_overlay=write_overlay
    )
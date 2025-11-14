import cv2
import mediapipe as mp
import numpy as np

def extract_pose(video_path):
    """
    Extract full body pose data from video for accurate swing analysis.
    
    Returns list of dicts with:
    - Wrist positions (left/right)
    - Shoulder positions (for body rotation reference)
    - Hip positions (for lower body rotation)
    - Elbow positions (for arm path)
    - Frame number
    """
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] Video properties:")
    print(f"  - FPS: {fps}")
    print(f"  - Resolution: {frame_width}x{frame_height}")
    print(f"  - Total frames: {total_frames}")
    
    # Setup video writer for overlay
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('overlay.mp4', fourcc, fps, (frame_width, frame_height))
    
    pose_data = []
    frame_number = 0
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose:
        
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                print(f"[INFO] Finished processing at frame {frame_number}")
                break
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Extract key body parts
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                
                # Store all data
                frame_data = {
                    'frame': frame_number,
                    
                    # Wrists (for club tracking)
                    'left_wrist_x': left_wrist.x,
                    'left_wrist_y': left_wrist.y,
                    'left_wrist_z': left_wrist.z,
                    'right_wrist_x': right_wrist.x,
                    'right_wrist_y': right_wrist.y,
                    'right_wrist_z': right_wrist.z,
                    
                    # Shoulders (for body rotation reference)
                    'left_shoulder_x': left_shoulder.x,
                    'left_shoulder_y': left_shoulder.y,
                    'left_shoulder_z': left_shoulder.z,
                    'right_shoulder_x': right_shoulder.x,
                    'right_shoulder_y': right_shoulder.y,
                    'right_shoulder_z': right_shoulder.z,
                    
                    # Hips (for lower body rotation)
                    'left_hip_x': left_hip.x,
                    'left_hip_y': left_hip.y,
                    'left_hip_z': left_hip.z,
                    'right_hip_x': right_hip.x,
                    'right_hip_y': right_hip.y,
                    'right_hip_z': right_hip.z,
                    
                    # Elbows (for arm path)
                    'left_elbow_x': left_elbow.x,
                    'left_elbow_y': left_elbow.y,
                    'left_elbow_z': left_elbow.z,
                    'right_elbow_x': right_elbow.x,
                    'right_elbow_y': right_elbow.y,
                    'right_elbow_z': right_elbow.z,
                }
                
                pose_data.append(frame_data)
                
                # Draw pose on frame
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Draw key points with labels
                h, w = frame.shape[:2]
                
                # Wrists (Yellow)
                cv2.circle(frame, (int(left_wrist.x * w), int(left_wrist.y * h)), 8, (0, 255, 255), -1)
                cv2.circle(frame, (int(right_wrist.x * w), int(right_wrist.y * h)), 8, (0, 255, 255), -1)
                
                # Shoulders (Green)
                cv2.circle(frame, (int(left_shoulder.x * w), int(left_shoulder.y * h)), 8, (0, 255, 0), -1)
                cv2.circle(frame, (int(right_shoulder.x * w), int(right_shoulder.y * h)), 8, (0, 255, 0), -1)
                
                # Draw shoulder line (body rotation reference)
                cv2.line(frame, 
                        (int(left_shoulder.x * w), int(left_shoulder.y * h)),
                        (int(right_shoulder.x * w), int(right_shoulder.y * h)),
                        (0, 255, 0), 3)
                
                # Add frame number
                cv2.putText(frame, f'Frame: {frame_number}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            else:
                # No pose detected this frame
                pose_data.append({
                    'frame': frame_number,
                    'left_wrist_x': None, 'left_wrist_y': None, 'left_wrist_z': None,
                    'right_wrist_x': None, 'right_wrist_y': None, 'right_wrist_z': None,
                    'left_shoulder_x': None, 'left_shoulder_y': None, 'left_shoulder_z': None,
                    'right_shoulder_x': None, 'right_shoulder_y': None, 'right_shoulder_z': None,
                    'left_hip_x': None, 'left_hip_y': None, 'left_hip_z': None,
                    'right_hip_x': None, 'right_hip_y': None, 'right_hip_z': None,
                    'left_elbow_x': None, 'left_elbow_y': None, 'left_elbow_z': None,
                    'right_elbow_x': None, 'right_elbow_y': None, 'right_elbow_z': None,
                })
            
            # Write frame to output video
            out.write(frame)
            
            frame_number += 1
            
            # Progress update
            if frame_number % 30 == 0:
                print(f"[INFO] Processed {frame_number}/{total_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"[INFO] Extraction complete! Processed {len(pose_data)} frames")
    print(f"[INFO] Overlay video saved as: overlay.mp4")
    
    return pose_data
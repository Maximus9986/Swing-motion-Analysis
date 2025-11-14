import cv2
import mediapipe as mp

def extract_pose(video_path):
    """
    Extract wrist pose data from video.
    Returns list of dicts with x, y, z coordinates and frame number.
    """
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Processing video: {fps} fps, {frame_width}x{frame_height}")
    
    # Setup video writer for overlay
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('overlay.mp4', fourcc, fps, (frame_width, frame_height))
    
    pose_data = []
    frame_number = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                break
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Extract left wrist (for right-handed golfer)
                wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                
                pose_data.append({
                    'frame': frame_number,
                    'x': wrist.x,
                    'y': wrist.y,
                    'z': wrist.z
                })
                
                # Draw pose on frame
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
            else:
                # No pose detected
                pose_data.append({
                    'frame': frame_number,
                    'x': None,
                    'y': None,
                    'z': None
                })
            
            out.write(frame)
            frame_number += 1
    
    cap.release()
    out.release()
    
    print(f"[INFO] Extracted {len(pose_data)} frames")
    return pose_data
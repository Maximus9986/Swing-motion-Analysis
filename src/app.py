# python -m streamlit run app.py
# http://localhost:8501

import streamlit as st
import pandas as pd
import os
import tempfile
import cv2
from PIL import Image
import numpy as np

from pose_tracking import extract_pose
from swing_analysis import analyze_swing, print_analysis_results
from visualisation import (
    plot_wrist_timeline, 
    plot_hand_path, 
    plot_tempo, 
    plot_speed_profile,
    plot_overall_score
)


def extract_phase_frames(video_path, df):
    """Extract frames at detected swing phases."""
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    # Use address_idx if available, otherwise backswing_start_idx
    address_frame = int(df['address_idx'].iloc[0]) if 'address_idx' in df.columns else int(df['backswing_start_idx'].iloc[0])
    
    phases = {
        'Address': address_frame,
        'Top of Backswing': int(df['backswing_top_idx'].iloc[0]),
        'Impact': int(df['impact_idx'].iloc[0]),
        'Finish': int(df['finish_idx'].iloc[0])
    }
    
    frames = {}
    
    for phase_name, frame_idx in phases.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[phase_name] = {
                'image': Image.fromarray(frame_rgb),
                'frame_idx': frame_idx
            }
    
    cap.release()
    return frames


def extract_phase_frames_with_skeleton(video_path, df):
    """Extract frames at detected swing phases with skeleton overlay."""
    import mediapipe as mp
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    # Use address_idx if available, otherwise backswing_start_idx
    address_frame = int(df['address_idx'].iloc[0]) if 'address_idx' in df.columns else int(df['backswing_start_idx'].iloc[0])
    
    phases = {
        'Start of backswing': address_frame,
        'Top of Backswing': int(df['backswing_top_idx'].iloc[0]),
        'Impact': int(df['impact_idx'].iloc[0]),
        'Finish': int(df['finish_idx'].iloc[0])
    }
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    frames = {}
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    ) as pose:
        
        for phase_name, frame_idx in phases.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                # Draw skeleton
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                    )
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[phase_name] = {
                    'image': Image.fromarray(frame_rgb),
                    'frame_idx': frame_idx
                }
    
    cap.release()
    return frames


st.set_page_config(page_title="Golf Swing Analyzer", layout="wide")

st.markdown("""
<div style="text-align:center;">
    <h1 style="font-size:42px;">🏌️ Golf Swing Analyzer</h1>
</div>
""", unsafe_allow_html=True)

# Ensure Data folder exists
os.makedirs(os.path.join("..", "Data"), exist_ok=True)
DATA_DIR = os.path.abspath(os.path.join("..", "Data"))

# =====================
# CLUB SELECTION
# =====================
st.sidebar.header("⚙️ Settings")
club_type = st.sidebar.radio(
    "Select Club Type",
    options=["Iron", "Driver"],
    index=0,
    help="Driver: Impact detected after lowest point (hitting up)\nIron: Impact at address level (hitting down)"
)

st.sidebar.markdown("""
---
### Club Type Info:
- **Iron**: Ball struck with descending blow
- **Driver**: Ball struck on the upswing (teed up)
""")

uploaded = st.file_uploader("Upload your swing video (MP4)", type=["mp4", "mov", "avi"])

if uploaded:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.flush()
    player_video_path = tfile.name

    st.subheader("📹 Uploaded Video")
    st.video(player_video_path)

    st.info(f"⏳ Analyzing swing with **{club_type}** settings... this may take a moment")
    overlay_path = os.path.join(DATA_DIR, "overlay.mp4")

    pose_rows = extract_pose(
        player_video_path,
        output_path=overlay_path,
        write_overlay=True,
        use_3d=False   
    )

    if not pose_rows:
        st.error("Pose extraction failed or video could not be opened.")
        st.stop()

    df = pd.DataFrame(pose_rows)
    
    # Pass club type to analyze_swing
    df = analyze_swing(df, club_type=club_type.lower())

    # Save CSV
    csv_out = os.path.join(DATA_DIR, "player_swing_analysis.csv")
    df.to_csv(csv_out, index=False)

    st.success("✅ Analysis complete!")

    # Data type indicator
    has_3d = df['has_3d_data'].iloc[0] if 'has_3d_data' in df.columns else False
    st.info(f"📊 Using {'3D pose data' if has_3d else '2D pose data (MediaPipe)'} | Club: **{club_type}**")

    # =====================
    # RESULTS SECTION
    # =====================
    st.header("🎯 Swing Analysis Results")

    # Overall Score (prominent display)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        score = df['overall_score'].iloc[0]
        rating = df['overall_rating'].iloc[0]
        
        if score >= 85:
            score_color = "green"
        elif score >= 70:
            score_color = "blue"
        elif score >= 55:
            score_color = "orange"
        else:
            score_color = "red"
        
        st.markdown(f"""
        <div style="text-align:center; padding:20px; background-color:#f0f2f6; border-radius:10px;">
            <h1 style="color:{score_color}; margin:0;">{score}/100</h1>
            <h3 style="margin:0;">{rating}</h3>
            <p style="margin:5px 0 0 0; color:gray;">Club: {club_type}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # =====================
    # PHASE FRAMES DISPLAY
    # =====================
    st.subheader("🎬 Swing Phases Visualization")
    
    show_skeleton = st.checkbox("Show pose skeleton overlay", value=False)
    
    if show_skeleton:
        phase_frames = extract_phase_frames_with_skeleton(player_video_path, df)
    else:
        phase_frames = extract_phase_frames(player_video_path, df)
    
    if phase_frames:
        cols = st.columns(4)
        
        for i, (phase_name, frame_data) in enumerate(phase_frames.items()):
            with cols[i]:
                st.image(
                    frame_data['image'], 
                    caption=f"{phase_name}\nFrame {frame_data['frame_idx']}", 
                    use_container_width=True
                )
    else:
        st.warning("Could not extract phase frames from video.")

    st.markdown("---")
        
    # Phase Detection
    st.subheader("📍 Phase Detection")
    col1, col2, col3, col4 = st.columns(4)
    
    address_frame = int(df['address_idx'].iloc[0]) if 'address_idx' in df.columns else int(df['backswing_start_idx'].iloc[0])
    
    with col1:
        st.metric("Address", f"Frame {address_frame}")
    with col2:
        st.metric("Top of Backswing", f"Frame {int(df['backswing_top_idx'].iloc[0])}")
    with col3:
        st.metric("Impact", f"Frame {int(df['impact_idx'].iloc[0])}")
        if 'impact_raw_idx' in df.columns:
            st.caption(f"(Raw: {int(df['impact_raw_idx'].iloc[0])})")
    with col4:
        st.metric("Finish", f"Frame {int(df['finish_idx'].iloc[0])}")

    st.markdown("---")

    # Metrics in columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⏱️ Tempo")
        tempo_ratio = df['tempo_ratio'].iloc[0]
        st.metric("Tempo Ratio", f"{tempo_ratio}:1")
        st.write(f"Backswing: {int(df['backswing_frames'].iloc[0])} frames")
        st.write(f"Downswing: {int(df['downswing_frames'].iloc[0])} frames")
        
        if 2.0 <= tempo_ratio <= 3.5:
            st.success("✅ Good tempo (tour average is ~3:1)")
        elif tempo_ratio < 2.0:
            st.warning("⚠️ Quick backswing - try slowing down")
        else:
            st.warning("⚠️ Slow downswing - try accelerating")

        st.subheader("🖐️ Hand Path")
        st.metric("Steepness Ratio", f"{df['hand_path_steepness'].iloc[0]:.2f}")
        st.write(f"Classification: **{df['hand_path_label'].iloc[0]}**")

    with col2:
        st.subheader("💪 Arm Extension")
        if df['elbow_angle_impact'].iloc[0] is not None:
            st.metric("Elbow Angle at Impact", f"{df['elbow_angle_impact'].iloc[0]:.1f}°")
            st.write(f"Classification: **{df['arm_extension_label'].iloc[0]}**")
        else:
            st.write("Data not available")

        st.subheader("⚡ Speed Timing")
        st.write(f"Max Speed Frame: {int(df['max_speed_frame'].iloc[0])}")
        st.write(f"Impact Frame: {int(df['impact_idx'].iloc[0])}")
        st.write(f"Assessment: **{df['speed_timing'].iloc[0]}**")

    st.markdown("---")

    # Visualizations
    st.header("📈 Visualizations")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Wrist Timeline", 
        "Hand Path", 
        "Tempo", 
        "Speed Profile",
        "Score Breakdown"
    ])

    with tab1:
        fig1 = plot_wrist_timeline(df)
        st.pyplot(fig1)

    with tab2:
        fig2 = plot_hand_path(df)
        st.pyplot(fig2)

    with tab3:
        fig3 = plot_tempo(df)
        st.pyplot(fig3)

    with tab4:
        fig4 = plot_speed_profile(df)
        st.pyplot(fig4)

    with tab5:
        fig5 = plot_overall_score(df)
        st.pyplot(fig5)

    # Download CSV
    st.markdown("---")
    with open(csv_out, "rb") as fh:
        st.download_button(
            "📥 Download Analysis CSV", 
            data=fh, 
            file_name="player_swing_analysis.csv", 
            mime="text/csv"
        )

    st.balloons()

else:
    st.info("👆 Upload a swing video to get started.")
    
    st.markdown(f"""
    ### How it works:
    1. **Select club type** in the sidebar (Iron or Driver)
    2. Upload a video of your golf swing (side view)
    3. The app extracts pose landmarks
    4. Key swing phases are detected automatically
    5. You get metrics on tempo, hand path, arm extension, and speed timing
    
    ### Club Type Differences:
    | | Iron | Driver |
    |---|---|---|
    | **Impact** | At address level (hitting down) | After lowest point (hitting up) |
    | **Hand Path** | Steeper is better | Shallower is better |
    | **Ball Position** | Middle of stance | Forward in stance |
    
    ### Tips for best results:
    - Use a side-on camera angle
    - Ensure full body is visible throughout the swing
    - Good lighting helps with pose detection
    - Make sure there is an address period
    """)
# python -m streamlit run app.py
# http://localhost:8501

import streamlit as st
import pandas as pd
import os
import tempfile
import cv2
from PIL import Image

from pose_tracking import extract_pose
from swing_analysis import analyze_swing
from visualisation import (
    plot_wrist_timeline,
    plot_hand_path,
    plot_tempo,
    plot_speed_profile,
    plot_overall_score
)

# Fixed club type (no user selection)
DEFAULT_CLUB_TYPE = "iron"


def extract_phase_frames(video_path, df):
    """Extract frames at detected swing phases."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    address_frame = int(df["address_idx"].iloc[0]) if "address_idx" in df.columns else 0
    top_frame = int(df["backswing_top_idx"].iloc[0]) if "backswing_top_idx" in df.columns else address_frame
    impact_frame = int(df["impact_idx"].iloc[0]) if "impact_idx" in df.columns else top_frame
    finish_frame = int(df["finish_idx"].iloc[0]) if "finish_idx" in df.columns else (len(df) - 1)

    phases = {
        "Address": address_frame,
        "Top of Backswing": int(df["backswing_top_idx"].iloc[0]),
        "Impact": int(df["impact_idx"].iloc[0]),
        "Finish": int(df["finish_idx"].iloc[0]),
    }

    frames = {}
    for phase_name, frame_idx in phases.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[phase_name] = {"image": Image.fromarray(frame_rgb), "frame_idx": frame_idx}

    cap.release()
    return frames


def extract_phase_frames_with_skeleton(video_path, df):
    """Extract frames at detected swing phases with skeleton overlay."""
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    address_frame = int(df["address_idx"].iloc[0]) if "address_idx" in df.columns else int(df["backswing_start_idx"].iloc[0])

    phases = {
        "Start of backswing": address_frame,
        "Top of Backswing": int(df["backswing_top_idx"].iloc[0]),
        "Impact": int(df["impact_idx"].iloc[0]),
        "Finish": int(df["finish_idx"].iloc[0]),
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

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[phase_name] = {"image": Image.fromarray(frame_rgb), "frame_idx": frame_idx}

    cap.release()
    return frames


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Golf Swing Analyzer", layout="wide")

st.markdown(
    """
<div style="text-align:center;">
    <h1 style="font-size:42px;">🏌️ Golf Swing Analyzer</h1>
</div>
""",
    unsafe_allow_html=True
)

# Ensure Data folder exists
os.makedirs(os.path.join("..", "Data"), exist_ok=True)
DATA_DIR = os.path.abspath(os.path.join("..", "Data"))

uploaded = st.file_uploader("Upload your swing video (MP4)", type=["mp4", "mov", "avi"])

if uploaded:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.flush()
    player_video_path = tfile.name

    st.subheader("📹 Uploaded Video")
    st.video(player_video_path)

    st.info("⏳ Analyzing swing... this may take a moment")

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

    # Analyze swing with fixed club type
    df = analyze_swing(df, club_type=DEFAULT_CLUB_TYPE)

    # Save CSV
    csv_out = os.path.join(DATA_DIR, "player_swing_analysis.csv")
    df.to_csv(csv_out, index=False)

    st.success("✅ Analysis complete!")

    # Data type indicator
    has_3d = df["has_3d_data"].iloc[0] if "has_3d_data" in df.columns else False
    st.info(f"📊 Using {'3D pose data' if has_3d else '2D pose data (MediaPipe)'}")

    # ----------------------------
    # RESULTS SECTION
    # ----------------------------
    st.header("🎯 Swing Analysis Results")

    # Overall Score (prominent display)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        score = int(df["overall_score"].iloc[0]) if "overall_score" in df.columns else 0
        rating = df["overall_rating"].iloc[0] if "overall_rating" in df.columns else "N/A"

        if score >= 85:
            score_color = "green"
        elif score >= 70:
            score_color = "blue"
        elif score >= 55:
            score_color = "orange"
        else:
            score_color = "red"

        st.markdown(
            f"""
        <div style="text-align:center; padding:20px; background-color:#f0f2f6; border-radius:10px;">
            <h1 style="color:{score_color}; margin:0;">{score}/100</h1>
            <h3 style="margin:0;">{rating}</h3>
        </div>
        """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ----------------------------
    # PHASE FRAMES DISPLAY
    # ----------------------------
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
                    frame_data["image"],
                    caption=f"{phase_name}\nFrame {frame_data['frame_idx']}",
                    use_container_width=True
                )
    else:
        st.warning("Could not extract phase frames from video.")

    st.markdown("---")

    # Phase Detection
    st.subheader("📍 Phase Detection")
    col1, col2, col3, col4 = st.columns(4)

    address_frame = int(df["address_idx"].iloc[0]) if "address_idx" in df.columns else int(df["backswing_start_idx"].iloc[0])

    with col1:
        st.metric("Address", f"Frame {address_frame}")
    with col2:
        st.metric("Top of Backswing", f"Frame {int(df['backswing_top_idx'].iloc[0])}")
    with col3:
        st.metric("Impact", f"Frame {int(df['impact_idx'].iloc[0])}")
        if "impact_raw_idx" in df.columns:
            st.caption(f"(Raw: {int(df['impact_raw_idx'].iloc[0])})")
    with col4:
        st.metric("Finish", f"Frame {int(df['finish_idx'].iloc[0])}")

    st.markdown("---")

    # Metrics in columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⏱️ Tempo")
        tempo_ratio = df["tempo_ratio"].iloc[0] if "tempo_ratio" in df.columns else None
        st.metric("Tempo Ratio", f"{tempo_ratio}:1" if tempo_ratio is not None else "N/A")

        if "backswing_frames" in df.columns:
            st.write(f"Backswing: {int(df['backswing_frames'].iloc[0])} frames")
        if "downswing_frames" in df.columns:
            st.write(f"Downswing: {int(df['downswing_frames'].iloc[0])} frames")

        if tempo_ratio is not None:
            if 2.0 <= float(tempo_ratio) <= 3.5:
                st.success("✅ Good tempo (tour average is ~3:1)")
            elif float(tempo_ratio) < 2.0:
                st.warning("⚠️ Quick backswing - try slowing down")
            else:
                st.warning("⚠️ Slow downswing - try accelerating")

        st.subheader("🖐️ Hand Path")
        if "hand_path_steepness" in df.columns:
            st.metric("Steepness Ratio", f"{float(df['hand_path_steepness'].iloc[0]):.2f}")
        if "hand_path_label" in df.columns:
            st.write(f"Classification: **{df['hand_path_label'].iloc[0]}**")

    with col2:
        st.subheader("💪 Arm Extension")
        if "elbow_angle_impact" in df.columns and pd.notna(df["elbow_angle_impact"].iloc[0]):
            st.metric("Elbow Angle at Impact", f"{float(df['elbow_angle_impact'].iloc[0]):.1f}°")
            st.write(f"Classification: **{df['arm_extension_label'].iloc[0]}**")
        else:
            st.write("Data not available")

        st.subheader("⚡ Speed Timing")
        if "max_speed_frame" in df.columns:
            st.write(f"Max Speed Frame: {int(df['max_speed_frame'].iloc[0])}")
        if "impact_idx" in df.columns:
            st.write(f"Impact Frame: {int(df['impact_idx'].iloc[0])}")
        if "speed_timing" in df.columns:
            st.write(f"Assessment: **{df['speed_timing'].iloc[0]}**")

    st.markdown("---")

    # ----------------------------
    # Visualizations
    # ----------------------------
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

    st.markdown(
        """
### How it works:
1. Upload a video of your golf swing (side view)
2. The app extracts pose landmarks
3. Key swing phases are detected automatically
4. You get metrics on tempo, hand path, arm extension, and speed timing

### Tips for best results:
- Use a side-on camera angle
- Ensure full body is visible throughout the swing
- Good lighting helps with pose detection
- Make sure there is an address period
"""
    )

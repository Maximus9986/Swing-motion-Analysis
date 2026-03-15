# python -m streamlit run app.py
# http://localhost:8501
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import hashlib
from PIL import Image

from pose_tracking import extract_pose
from club_tracking import extract_clubhead_y_from_yolo
from swing_analysis import analyze_swing
from visualisation import (
    plot_wrist_timeline,
    plot_tempo,
    plot_speed_profile,
    plot_overall_score
)

# ✅ Windows path: use raw string
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLUB_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# ----------------------------
# Caching helpers
# ----------------------------
def _file_hash(uploaded_file) -> str:
    """Stable key so Streamlit cache/session resets when a new file is uploaded."""
    data = uploaded_file.getvalue()
    return hashlib.md5(data).hexdigest()

@st.cache_data(show_spinner=True)
def run_pose(video_path: str, handed: str = "right"):
    """Run MediaPipe pose once per video + handedness combo."""
    pose_rows = extract_pose(
        video_path,
        output_path=None,
        write_overlay=False,
        use_3d=False,
        handed=handed
    )
    return pd.DataFrame(pose_rows)

@st.cache_data(show_spinner=False)
def run_analysis(df: pd.DataFrame, fps: float):
    """Run swing analysis once per pose dataframe."""
    return analyze_swing(df, fps=fps, debug=False)


# ----------------------------
# Frame extraction helpers
# ----------------------------
def extract_phase_frames(video_path, df):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    def get_idx(col, default=0):
        try:
            return int(df[col].iloc[0]) if col in df.columns else int(default)
        except Exception:
            return int(default)

    address_frame = get_idx("address_idx", 0)

    phases = {
        "Address": address_frame,
        "Top of Backswing": get_idx("backswing_top_idx", address_frame),
        "Impact": get_idx("impact_idx", address_frame),
        "Finish": get_idx("finish_idx", len(df) - 1),
    }

    frames = {}
    for phase_name, frame_idx in phases.items():
        frame_idx = max(0, frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[phase_name] = {"image": Image.fromarray(frame_rgb), "frame_idx": frame_idx}

    cap.release()
    return frames


def extract_phase_frames_with_skeleton(video_path, df):
    import mediapipe as mp
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    def get_idx(col, default=0):
        try:
            return int(df[col].iloc[0]) if col in df.columns else int(default)
        except Exception:
            return int(default)

    address_frame = get_idx("address_idx", 0)

    phases = {
        "Address": address_frame,
        "Top of Backswing": get_idx("backswing_top_idx", address_frame),
        "Impact": get_idx("impact_idx", address_frame),
        "Finish": get_idx("finish_idx", len(df) - 1),
    }

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    frames = {}
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
        for phase_name, frame_idx in phases.items():
            frame_idx = max(0, frame_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

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

DATA_DIR = os.path.abspath(os.path.join("..", "Data"))
os.makedirs(DATA_DIR, exist_ok=True)

uploaded = st.file_uploader("Upload your swing video (MP4)", type=["mp4", "mov", "avi"])

# Fix 3: Handedness toggle in sidebar
handed = st.sidebar.radio(
    "Golfer handedness",
    options=["right", "left"],
    index=0,
    help="Select the golfer's dominant hand. This determines which arm is tracked as the lead arm."
)

if uploaded:
    video_key = _file_hash(uploaded)

    # Reset stored results if a different video OR handedness is selected
    cache_key = f"{video_key}_{handed}"
    if st.session_state.get("cache_key") != cache_key:
        st.session_state.cache_key = cache_key
        st.session_state.video_key = video_key
        st.session_state.analysis_df = None
        st.session_state.pose_df = None
        st.session_state.club_df = None

    # Save uploaded video ONCE per new upload
    if "player_video_path" not in st.session_state or st.session_state.get("player_video_key") != video_key:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.getvalue())
        tfile.flush()
        st.session_state.player_video_path = tfile.name
        st.session_state.player_video_key = video_key

    player_video_path = st.session_state.player_video_path

    st.subheader("📹 Uploaded Video")
    st.video(player_video_path)

    st.info("⏳ Analyzing swing... this may take a moment")

    # Get FPS once
    cap = cv2.VideoCapture(player_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # Pose extraction (cached by video + handedness)
    if st.session_state.pose_df is None:
        st.session_state.pose_df = run_pose(player_video_path, handed=handed)

    df = st.session_state.pose_df.copy()
    if "frame" not in df.columns:
        df["frame"] = np.arange(len(df))

    # YOLO (cached via session_state)
    if st.session_state.club_df is None:
        try:
            st.session_state.club_df = extract_clubhead_y_from_yolo(
                player_video_path,
                CLUB_MODEL_PATH,
                conf=0.25,
                clubhead_kpt_idx=0
            )
        except Exception as e:
            st.session_state.club_df = None
            st.warning(f"YOLO club tracking failed (wrist-only impact). Error: {e}")

    if st.session_state.club_df is not None:
        club_df = st.session_state.club_df
        df = df.merge(
            club_df[["frame", "clubhead_y", "clubhead_valid", "clubhead_y_smooth"]],
            on="frame",
            how="left"
        )
        st.caption(f"YOLO detections: {int(df['clubhead_valid'].sum())}/{len(df)} frames")

    # Analysis (cached)
    if st.session_state.analysis_df is None:
        st.session_state.analysis_df = run_analysis(df, fps)

    df = st.session_state.analysis_df.copy()

    # Show whether YOLO was used
    if "impact_method" in df.columns:
        st.caption(f"Impact method: {df['impact_method'].iloc[0]}")

    # Save CSV
    csv_out = os.path.join(DATA_DIR, "player_swing_analysis.csv")
    df.to_csv(csv_out, index=False)

    st.success("✅ Analysis complete!")

    # ----------------------------
    # Results
    # ----------------------------
    st.header("🎯 Swing Analysis Results")

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
    # Phase frames
    # ----------------------------
    st.subheader("🎬 Swing Phases Visualization")
    show_skeleton = st.checkbox("Show pose skeleton overlay", value=False)

    phase_frames = extract_phase_frames_with_skeleton(player_video_path, df) if show_skeleton else extract_phase_frames(player_video_path, df)
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

    # ----------------------------
    # Phase indices
    # ----------------------------
    st.subheader("📍 Phase Detection")
    c1, c2, c3, c4 = st.columns(4)

    def gi(name, default=0):
        return int(df[name].iloc[0]) if name in df.columns else int(default)

    with c1:
        st.metric("Address", f"Frame {gi('address_idx')}")
    with c2:
        st.metric("Top of Backswing", f"Frame {gi('backswing_top_idx')}")
    with c3:
        st.metric("Impact", f"Frame {gi('impact_idx')}")
        if "impact_raw_idx" in df.columns:
            st.caption(f"(Raw wrist: {gi('impact_raw_idx')})")
        if "impact_method" in df.columns:
            st.caption(f"(Method: {df['impact_method'].iloc[0]})")
    with c4:
        st.metric("Finish", f"Frame {gi('finish_idx')}")

    st.markdown("---")

    # ----------------------------
    # Metrics
    # ----------------------------
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("⏱️ Tempo")
        tempo_ratio = df["tempo_ratio"].iloc[0] if "tempo_ratio" in df.columns else None
        st.metric("Tempo Ratio", f"{tempo_ratio}:1" if tempo_ratio is not None else "N/A")
        st.write(f"Backswing: {gi('backswing_frames')} frames")
        st.write(f"Downswing: {gi('downswing_frames')} frames")

        if tempo_ratio is not None:
            if 2.0 <= float(tempo_ratio) <= 3.5:
                st.success("✅ Good tempo (tour average is ~3:1)")
            elif float(tempo_ratio) < 2.0:
                st.warning("⚠️ Quick backswing - try slowing down")
            else:
                st.warning("⚠️ Slow downswing - try accelerating")

        st.subheader("🏋️ Early Extension (Side-view proxy)")
        if "early_extension_score" in df.columns:
            st.metric("Early Extension Score", f"{gi('early_extension_score')}/100")
            st.write(f"Assessment: **{df['early_extension_label'].iloc[0]}**")
        else:
            st.write("N/A")

    with c2:
        st.subheader("💪 Arm Extension")
        if "elbow_angle_impact" in df.columns and pd.notna(df["elbow_angle_impact"].iloc[0]):
            st.metric("Elbow Angle at Impact", f"{float(df['elbow_angle_impact'].iloc[0]):.1f}°")
            st.write(f"Classification: **{df['arm_extension_label'].iloc[0]}**")
        else:
            st.write("Data not available")

        st.subheader("⚡ Speed Timing")
        if "max_speed_frame" in df.columns:
            st.write(f"Max Speed Frame: {gi('max_speed_frame')}")
        st.write(f"Impact Frame: {gi('impact_idx')}")
        if "speed_timing" in df.columns:
            st.write(f"Assessment: **{df['speed_timing'].iloc[0]}**")

    st.markdown("---")

    # ----------------------------
    # Plots
    # ----------------------------
    st.header("📈 Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(["Wrist Timeline", "Tempo", "Speed Profile", "Score Breakdown"])

    with tab1:
        st.pyplot(plot_wrist_timeline(df))
    with tab2:
        st.pyplot(plot_tempo(df))
    with tab3:
        st.pyplot(plot_speed_profile(df))
    with tab4:
        st.pyplot(plot_overall_score(df))

    st.markdown("---")
    with open(csv_out, "rb") as fh:
        st.download_button(
            "📥 Download Analysis CSV",
            data=fh,
            file_name="player_swing_analysis.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Upload a swing video to get started.")
    st.markdown("""
### How it works:
1. Upload a side-view video of your golf swing
2. Select your handedness in the sidebar (right or left-handed)
3. The app extracts pose landmarks (MediaPipe)
4. Key swing phases are detected automatically
5. You get metrics: tempo, arm extension, speed timing, and early extension risk

### Tips for best results:
- Keep the camera fixed (tripod if possible)
- Ensure full body is visible throughout the swing
- Good lighting helps pose tracking
- Include a short address setup before swinging
""")
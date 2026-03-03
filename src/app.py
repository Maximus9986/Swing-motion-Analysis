# python -m streamlit run app.py
# http://localhost:8501
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
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
CLUB_MODEL_PATH = r"C:\Users\User\FYP\Swing-motion-Analysis\src\models\best.pt"

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

if uploaded:
    # 1) Save uploaded video ONCE
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.flush()
    player_video_path = tfile.name

    st.subheader("📹 Uploaded Video")
    st.video(player_video_path)

    st.info("⏳ Analyzing swing... this may take a moment")

    # 2) Pose extraction ONCE
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

    # Ensure frame index exists (needed for merge)
    if "frame" not in df.columns:
        df["frame"] = np.arange(len(df))

    # 3) YOLO club tracking ONCE
    try:
        club_df = extract_clubhead_y_from_yolo(
            player_video_path,
            CLUB_MODEL_PATH,
            conf=0.25,
            clubhead_kpt_idx=0
        )

        df = df.merge(
            club_df[["frame", "clubhead_y", "clubhead_valid", "clubhead_y_smooth"]],
            on="frame",
            how="left"
        )

        st.caption(f"YOLO detections: {int(df['clubhead_valid'].sum())}/{len(df)} frames")
    except Exception as e:
        st.warning(f"YOLO club tracking failed (wrist-only impact). Error: {e}")

    # 4) Analyze ONCE (impact refinement happens inside analyze_swing)
    df = analyze_swing(df, debug=False)

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
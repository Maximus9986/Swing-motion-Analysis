# python -m streamlit run app.py
import streamlit as st
import pandas as pd
import os
import tempfile

from pose_tracking import extract_pose
from swing_analysis import analyze_swing
from visualisation import plot_wrist_timeline, plot_birds_eye, plot_tempo

st.set_page_config(page_title="Golf Swing Analyzer", layout="wide")

st.markdown("""
<div style="text-align:center;">
    <img src="https://media.giphy.com/media/Q67BhXCom9PepOhYZd/giphy.gif" width="120"><br>
    <h1 style="font-size:42px;">Golf Swing Analyzer</h1>
</div>
""", unsafe_allow_html=True)

# ensure Data folder
os.makedirs(os.path.join("..", "Data"), exist_ok=True)
DATA_DIR = os.path.abspath(os.path.join("..", "Data"))

uploaded = st.file_uploader("Upload your swing video (MP4)", type=["mp4","mov","avi"])

if uploaded:
    # Save uploaded file to a temp path
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.flush()
    player_video_path = tfile.name

    st.subheader("📹 Uploaded video")
    st.video(player_video_path)

    st.info("⏳ Extracting pose & generating overlay... this may take a bit")
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
    df = analyze_swing(df)

    # Save CSV
    csv_out = os.path.join(DATA_DIR, "player_swing_analysis.csv")
    df.to_csv(csv_out, index=False)

    st.success("✅ Analysis complete")

    # # Show overlay if exists
    # if os.path.exists(overlay_path):
    #     st.subheader("📹 Overlay video (pose)")
    #     with open(overlay_path, "rb") as vf:
    #         st.video(vf.read())

    # Show metrics
    st.subheader("🎯 Swing Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Frames", len(df))
    with col2:
        st.metric("Detection Rate", f"{df['wrist_x'].notna().sum()}/{len(df)}")
    with col3:
        st.metric("Path Quality (R²)", f"{df['path_quality'].iloc[0]:.3f}")

    st.markdown("**Path classification:** " + str(df["swing_path_label"].iloc[0]))
    st.markdown("**Path angle:** " + (f"{df['swing_path_angle'].iloc[0]:.2f}°" if pd.notna(df['swing_path_angle'].iloc[0]) else "N/A"))
    st.markdown("**Predicted Ball Flight:** " + str(df["ball_flight"].iloc[0])) 
    st.markdown("**Tempo (backswing/downswing frames)**: " + f"{df['backswing_frames'].iloc[0]} / {df['downswing_frames'].iloc[0]} (ratio {df['tempo_ratio'].iloc[0]})")

    # Plots
    st.subheader("📈 Wrist Timeline")
    fig1 = plot_wrist_timeline(df)
    st.pyplot(fig1)

    st.subheader("🛰 Bird's-eye downswing path (Z→X)")
    fig2 = plot_birds_eye(df)
    st.pyplot(fig2)

    st.subheader("⏱ Tempo")
    fig3 = plot_tempo(df)
    st.pyplot(fig3)

    # Download CSV
    with open(csv_out, "rb") as fh:
        st.download_button("📥 Download analysis CSV", data=fh, file_name="player_swing_analysis.csv", mime="text/csv")

    st.balloons()
else:
    st.info("Upload a swing video to get started.")

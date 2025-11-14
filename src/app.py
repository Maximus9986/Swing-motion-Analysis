import streamlit as st
import pandas as pd
import os

from pose_tracking import extract_pose
from swing_analysis import analyze_swing

st.title("🏌️ Golf Swing Analyzer")
st.write("Upload your swing video to analyze swing path.")

uploaded = st.file_uploader("Upload your swing video (MP4)", type=["mp4"])

if uploaded:
    # Save uploaded video
    player_video_path = "temp.mp4"
    with open(player_video_path, "wb") as f:
        f.write(uploaded.read())

    st.info("⏳ Extracting pose data, please wait...")
    
    # Show original video
    st.subheader("📹 Original Video")
    st.video(player_video_path)

    try:
        # Extract pose data
        pose_data = extract_pose(player_video_path)
        
        if not pose_data or len(pose_data) < 30:
            st.error("❌ Video too short or pose detection failed.")
            st.stop()
        
        # Convert to dataframe
        df = pd.DataFrame(pose_data)
        
        # Check if we have wrist data
        if "x" not in df.columns or df["x"].isna().all():
            st.error("❌ Could not detect wrist position in video.")
            st.stop()
        
        # Analyze swing
        df = analyze_swing(df)
        
        st.success("✅ Analysis completed!")
        
        # Show overlay video if exists
        st.markdown("---")
        
        if os.path.exists("overlay.mp4"):
            st.subheader("🎨 Video with Pose Detection")
            st.video("overlay.mp4")
            
            with open("overlay.mp4", "rb") as video_file:
                st.download_button(
                    label="📥 Download Overlay Video",
                    data=video_file,
                    file_name="swing_with_pose.mp4",
                    mime="video/mp4"
                )
        
        st.markdown("---")
        
        # Show data preview
        with st.expander("📊 View Raw Data"):
            st.dataframe(df.head(20))
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="swing_data.csv",
                mime="text/csv"
            )

        # ===== SWING PATH RESULTS =====
        st.markdown("---")
        st.subheader("🎯 Swing Path Analysis")
        
        if "swing_path_label" in df.columns and "swing_path_angle" in df.columns:
            path_label = df["swing_path_label"].iloc[0]
            angle = df["swing_path_angle"].iloc[0]
            ball_flight = df.get("ball_flight", pd.Series(["Unknown"])).iloc[0]
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Swing Path", path_label)
            
            with col2:
                st.metric("Path Angle", f"{angle:.2f}°")
            
            with col3:
                st.metric("Expected Ball Flight", ball_flight)
            
            # Quality metric
            if "path_quality" in df.columns:
                quality = df["path_quality"].iloc[0]
                st.metric("Path Quality (R²)", f"{quality:.3f}")
            
            # Interpretation
            st.markdown("---")
            st.subheader("📖 What This Means:")
            
            abs_angle = abs(angle)
            
            if abs_angle < 2:
                st.success(f"""
                ✅ **Excellent! Neutral Swing Path**
                
                Your club is traveling straight down the target line.
                Expected ball flight: Straight shots with minimal curve.
                """)
            elif angle > 0:
                st.info(f"""
                🔵 **In-to-Out Swing Path**
                
                Your club is traveling from inside to outside at impact.
                Expected ball flight: {ball_flight} (curves left for right-handed golfer)
                """)
            else:
                st.warning(f"""
                🔴 **Out-to-In Swing Path**
                
                Your club is traveling from outside to inside at impact.
                Expected ball flight: {ball_flight} (curves right for right-handed golfer)
                """)
        
        else:
            st.warning("⚠️ Swing path analysis not available.")
        
        # Additional metrics
        st.markdown("---")
        st.subheader("📊 Detection Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Frames", len(df))
        
        with col2:
            valid_frames = df["x"].notna().sum()
            st.metric("Frames with Pose", valid_frames)
        
        with col3:
            detection_rate = (valid_frames / len(df)) * 100
            st.metric("Detection Rate", f"{detection_rate:.1f}%")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        
        with st.expander("🔍 Show error details"):
            import traceback
            st.code(traceback.format_exc())
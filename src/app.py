import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from pose_tracking import extract_pose
from swing_analysis import analyze_swing

st.title("🏌️ Golf Swing Analyzer")
st.write("Upload your swing video to analyse wrist motion, swing path, and key features.")

uploaded = st.file_uploader("Upload your swing video (MP4)", type=["mp4"])

if uploaded:
    # Save uploaded video temporarily
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
        
        # Check if we got data
        if not pose_data or len(pose_data) < 30:
            st.error("❌ Video too short or pose detection failed.")
            st.info("💡 Tips:")
            st.write("- Video should be at least 1 second (30 frames)")
            st.write("- Ensure golfer is clearly visible")
            st.write("- Use side view angle")
            st.stop()
        
        # Convert to dataframe
        df = pd.DataFrame(pose_data)
        
        # Check if we have wrist data
        if "x" not in df.columns or df["x"].isna().all():
            st.error("❌ Could not detect wrist position in video.")
            st.info("💡 Tips:")
            st.write("- Ensure person is fully visible in frame")
            st.write("- Use good lighting")
            st.write("- Avoid busy backgrounds")
            st.write("- Use side view (perpendicular to target line)")
            st.stop()
        
        # Analyze swing
        df = analyze_swing(df)
        
        st.success("✅ Pose extraction completed!")
        
        # Show overlay video if it exists
        st.markdown("---")
        
        # Check for overlay video
        overlay_paths = ["overlay.mp4", "overlay_with_pose.mp4", "temp_overlay.mp4"]
        overlay_found = False
        
        for overlay_path in overlay_paths:
            if os.path.exists(overlay_path):
                file_size = os.path.getsize(overlay_path)
                if file_size > 0:
                    st.subheader("🎨 Video with Pose Detection")
                    st.video(overlay_path)
                    
                    # Download button
                    with open(overlay_path, "rb") as video_file:
                        st.download_button(
                            label="📥 Download Overlay Video",
                            data=video_file,
                            file_name="swing_with_pose.mp4",
                            mime="video/mp4"
                        )
                    overlay_found = True
                    break
        
        if not overlay_found:
            st.warning("⚠️ Overlay video was not created.")
            st.info("The analysis will continue, but you won't see the pose overlay.")
        
        st.markdown("---")
        
        # Show data preview
        with st.expander("📊 View Raw Data"):
            st.dataframe(df.head(20))
            
            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Data (CSV)",
                data=csv,
                file_name="swing_data.csv",
                mime="text/csv"
            )

        # -----------------
        # Swing Path Result
        # -----------------
        st.markdown("---")
        st.subheader("🎯 Swing Path Analysis")
        
        if "swing_path_label" in df.columns and "swing_path_angle" in df.columns:
            path_label = df["swing_path_label"].iloc[0]
            angle = df["swing_path_angle"].iloc[0]
            ball_flight = df.get("ball_flight", pd.Series(["Unknown"])).iloc[0]
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Swing Path**")
                st.markdown(f"<div style='font-size: 1.5rem; font-weight: 600; color: #0068C9;'>{path_label}</div>", 
                        unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Path Angle**")
                # Color code based on angle magnitude
                angle_color = "#FF4B4B" if abs(angle) > 8 else "#FFA500" if abs(angle) > 3 else "#00C851"
                st.markdown(f"<div style='font-size: 1.5rem; font-weight: 600; color: {angle_color};'>{angle:.2f}°</div>", 
                        unsafe_allow_html=True)
            
            with col3:
                st.markdown("**Expected Ball Flight**")
                st.markdown(f"<div style='font-size: 1.5rem; font-weight: 600; color: #0068C9;'>{ball_flight}</div>", 
                        unsafe_allow_html=True)
            
            # Quality metric
            if "path_quality" in df.columns:
                quality = df["path_quality"].iloc[0]
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric(
                        "Path Consistency (R²)",
                        f"{quality:.3f}",
                        help="How straight and consistent your swing path is (0-1, higher is better)"
                    )
            
            # Interpretation
            st.markdown("---")
            st.subheader("📖 What This Means:")
            
            abs_angle = abs(angle)
            
            if abs_angle < 2:
                # Straight shot
                st.success(f"""
                ✅ **Excellent! Neutral Swing Path**
                
                **Angle:** {angle:.2f}° (Very close to 0°)
                
                **What's happening:**
                Your club is traveling straight down the target line at impact.
                
                **Expected ball flight:** Straight shots with minimal curve
                
                **This is ideal!** Keep practicing to maintain this consistency.
                """)
            
            elif angle > 0:
                # In-to-Out (Draw/Hook)
                if abs_angle < 5:
                    severity = "slight"
                    icon = "🔵"
                    tips_line1 = "- This is a controlled draw shape - great for distance!"
                    tips_line2 = "- Keep face square to path to control the curve"
                elif abs_angle < 8:
                    severity = "moderate"
                    icon = "🔵"
                    tips_line1 = "- This is a controlled draw shape - great for distance!"
                    tips_line2 = "- Keep face square to path to control the curve"
                else:
                    severity = "strong"
                    icon = "🔴"
                    tips_line1 = "- Too much hook - try rotating shoulders more through impact"
                    tips_line2 = "- Check your grip - might be too strong"
                
                st.info(f"""
                {icon} **In-to-Out Swing Path ({severity})**
                
                **Angle:** {angle:.2f}°
                
                **What's happening:**
                Your club is traveling from inside the target line to outside at impact.
                
                **Expected ball flight:** {ball_flight} (curves left for right-handed golfer)
                
                **Tips:**
                {tips_line1}
                {tips_line2}
                """)
            
            else:
                # Out-to-In (Fade/Slice)
                if abs_angle < 5:
                    severity = "slight"
                    icon = "🟡"
                    tips_line1 = "- This is a controlled fade shape - good for accuracy!"
                    tips_line2 = "- Fade is safer than hook for most situations"
                elif abs_angle < 8:
                    severity = "moderate"
                    icon = "🟠"
                    tips_line1 = "- This is a controlled fade shape - good for accuracy!"
                    tips_line2 = "- Fade is safer than hook for most situations"
                else:
                    severity = "strong"
                    icon = "🔴"
                    tips_line1 = "- Too much slice - focus on swinging from the inside"
                    tips_line2 = "- Check your grip and make sure you are not coming over the top"
                
                st.warning(f"""
                {icon} **Out-to-In Swing Path ({severity})**
                
                **Angle:** {angle:.2f}°
                
                **What's happening:**
                Your club is traveling from outside the target line to inside at impact.
                
                **Expected ball flight:** {ball_flight} (curves right for right-handed golfer)
                
                **Tips:**
                {tips_line1}
                {tips_line2}
                """)
        
        else:
            st.warning("⚠️ Swing path analysis not available.")

        # -----------------
        # Wrist Trajectory Graph
        # -----------------
        st.markdown("---")
        st.subheader("📈 Wrist Trajectory Timeline")
        
        if "wrist_y_smooth" in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot trajectory
            ax.plot(df.index, df["wrist_y_smooth"], 
                label="Wrist Height", color="blue", linewidth=2)
            
            # Mark key points
            if len(df) > 0:
                if "backswing_peak_frame" in df.columns:
                    peak_idx = int(df["backswing_peak_frame"].iloc[0])
                else:
                    # Fallback to highest point if not detected
                    peak_idx = df["wrist_y_smooth"].idxmax()
                
                peak_value = df.loc[peak_idx, "wrist_y_smooth"]
                
                ax.axvline(peak_idx, color='green', linestyle='--', 
                        alpha=0.7, label='Top of Backswing (Transition)')
                ax.scatter(peak_idx, peak_value, color='green', 
                        s=100, zorder=5)
                
                # ✅ ALSO mark impact point
                # Find impact by speed spike after backswing
                if "wrist_speed" in df.columns:
                    search_end = min(peak_idx + 20, len(df) - 1)
                    search_window = df.loc[peak_idx:search_end]
                    if len(search_window) > 0:
                        impact_idx = search_window["wrist_speed"].idxmax()
                        if not pd.isna(impact_idx) and impact_idx in df.index:
                            impact_value = df.loc[impact_idx, "wrist_y_smooth"]
                            ax.axvline(impact_idx, color='red', linestyle='--', 
                                    alpha=0.7, label='Impact (Speed Peak)')
                            ax.scatter(impact_idx, impact_value, color='red', 
                                    s=150, marker='*', zorder=5, edgecolors='black')
            
            ax.set_xlabel("Frame Number", fontsize=12)
            ax.set_ylabel("Wrist Height (normalized)", fontsize=12)
            ax.set_title("Wrist Path During Swing", fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Save and offer download
            fig.savefig("wrist_trajectory.png", dpi=150, bbox_inches='tight')
            with open("wrist_trajectory.png", "rb") as img_file:
                st.download_button(
                    label="📥 Download Chart",
                    data=img_file,
                    file_name="wrist_trajectory.png",
                    mime="image/png"
                )
        else:
            st.warning("⚠️ Wrist trajectory data not available.")
        
        # Additional metrics
        st.markdown("---")
        st.subheader("📊 Additional Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Frames", len(df))
        
        with col2:
            valid_frames = df["x"].notna().sum()
            st.metric("Frames with Pose", valid_frames)
        
        with col3:
            detection_rate = (valid_frames / len(df)) * 100
            st.metric("Detection Rate", f"{detection_rate:.1f}%")
        
        if detection_rate < 70:
            st.warning("⚠️ Low pose detection rate. Consider re-recording with better lighting and clearer view of golfer.")
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        
        with st.expander("🔍 Show detailed error"):
            import traceback
            st.code(traceback.format_exc())
        
        st.info("💡 Common issues:")
        st.write("- Video format not supported (try converting to H.264 MP4)")
        st.write("- Poor video quality or lighting")
        st.write("- Golfer not fully visible in frame")
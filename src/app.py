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
        # Swing Path Result - UPDATED FOR NEW VERSION
        # -----------------
        st.markdown("---")
        st.subheader("🎯 Swing Path Analysis")
        
        if "swing_path_label" in df.columns and "swing_path_angle" in df.columns:
            path_label = df["swing_path_label"].iloc[0]
            angle = df["swing_path_angle"].iloc[0]
            ball_flight = df.get("ball_flight", pd.Series(["Unknown"])).iloc[0]
            
            # Display metrics in a cleaner way
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
            
            # Interpretation based on angle
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
                elif abs_angle < 8:
                    severity = "moderate"
                    icon = "🔵"
                else:
                    severity = "strong"
                    icon = "🔴"
                
                st.info(f"""
                {icon} **In-to-Out Swing Path ({severity})**
                
                **Angle:** {angle:.2f}°
                
                **What's happening:**
                Your club is traveling from inside the target line to outside at impact.
                
                **Expected ball flight:** {ball_flight} (curves left for right-handed golfer)
                
                **Tips:**
                {'- This is a controlled draw shape - great for distance!' if abs_angle < 8 else '- Too much hook - try rotating shoulders more through impact'}
                {'- Keep face square to path to control the curve' if abs_angle < 8 else '- Check your grip - might be too strong'}
                """)
            
            else:
                # Out-to-In (Fade/Slice)
                if abs_angle < 5:
                    severity = "slight"
                    icon = "🟡"
                elif abs_angle < 8:
                    severity = "moderate"
                    icon = "🟠"
                else:
                    severity = "strong"
                    icon = "🔴"
                
                st.warning(f"""
                {icon} **Out-to-In Swing Path ({severity})**
                
                **Angle:** {angle:.2f}°
                
                **What's happening:**
                Your club is traveling from outside the target line to inside at impact.
                
                **Expected ball flight:** {ball_flight} (curves right for right-handed golfer)
                
                **Tips:**
                {'- This is a controlled fade shape - good for accuracy!' if abs_angle < 8 else '- Too much slice - focus on swinging from the inside'}
                {'- Fade is safer than hook for most situations' if abs_angle < 8 else '- Check your grip and make sure you are not coming over the top'}
                """)
        
        else:
            st.warning("⚠️ Swing path analysis not available.")

        # -----------------
        # Wrist Trajectory Graph with ALL Swing Phases
        # -----------------
        st.markdown("---")
        st.subheader("📈 Swing Phases Analysis")
        
        if "wrist_y_smooth" in df.columns:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Plot main trajectory
            ax.plot(df.index, df["wrist_y_smooth"], 
                   label="Wrist Height", color="blue", linewidth=3)
            
            # Get phase markers
            if "backswing_peak_frame" in df.columns and "impact_frame" in df.columns:
                backswing_peak = int(df["backswing_peak_frame"].iloc[0])
                impact = int(df["impact_frame"].iloc[0])
                
                # Calculate additional phase points
                # Address/Setup: start of video
                address = 0
                
                # Backswing: frames from address to backswing peak
                backswing_mid = (address + backswing_peak) // 2
                
                # Downswing: frames from backswing peak to impact
                downswing_mid = (backswing_peak + impact) // 2
                
                # Follow-through: frames after impact
                follow_through_start = impact + 5
                follow_through_end = min(impact + 20, len(df) - 1)
                
                # ===== PHASE 1: ADDRESS/SETUP (Frame 0) =====
                address_y = df.loc[address, "wrist_y_smooth"]
                ax.axvline(address, color='purple', linestyle=':', linewidth=2, alpha=0.5)
                ax.scatter(address, address_y, color='purple', s=150, marker='o', 
                          zorder=10, edgecolors='black', linewidths=2, label='Setup/Address')
                ax.text(address, address_y, '  Setup', fontsize=10, verticalalignment='bottom')
                
                # ===== PHASE 2: BACKSWING PEAK (Green) =====
                backswing_y = df.loc[backswing_peak, "wrist_y_smooth"]
                ax.axvline(backswing_peak, color='green', linestyle='--', linewidth=2, alpha=0.7)
                ax.scatter(backswing_peak, backswing_y, color='green', s=200, marker='o', 
                          zorder=10, edgecolors='black', linewidths=2, label='Top of Backswing')
                ax.text(backswing_peak, backswing_y, '  Backswing Peak', fontsize=10, verticalalignment='top')
                
                # ===== PHASE 3: IMPACT (Red Star) =====
                impact_y = df.loc[impact, "wrist_y_smooth"]
                ax.axvline(impact, color='red', linestyle='--', linewidth=2, alpha=0.7)
                ax.scatter(impact, impact_y, color='red', s=300, marker='*', 
                          zorder=10, edgecolors='black', linewidths=2, label='Impact')
                ax.text(impact, impact_y, '  IMPACT', fontsize=11, fontweight='bold', 
                       verticalalignment='bottom', color='red')
                
                # ===== PHASE 4: FOLLOW-THROUGH (Orange) =====
                if follow_through_end < len(df):
                    follow_y = df.loc[follow_through_end, "wrist_y_smooth"]
                    ax.axvline(follow_through_end, color='orange', linestyle=':', linewidth=2, alpha=0.5)
                    ax.scatter(follow_through_end, follow_y, color='orange', s=150, marker='o', 
                              zorder=10, edgecolors='black', linewidths=2, label='Follow-Through')
                    ax.text(follow_through_end, follow_y, '  Follow Through', fontsize=10, 
                           verticalalignment='top')
                
                # ===== SHADE PHASE REGIONS =====
                # Backswing region (light green)
                ax.axvspan(address, backswing_peak, alpha=0.1, color='green', label='Backswing Phase')
                
                # Downswing region (light yellow)
                ax.axvspan(backswing_peak, impact, alpha=0.1, color='yellow', label='Downswing Phase')
                
                # Follow-through region (light orange)
                ax.axvspan(impact, len(df)-1, alpha=0.1, color='orange', label='Follow-Through Phase')
                
                # Add phase statistics
                st.markdown("---")
                st.subheader("⏱️ Swing Timing")
                
                fps = 30  # Assume 30 fps (you can get this from video metadata)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    backswing_frames = backswing_peak - address
                    backswing_time = backswing_frames / fps
                    st.metric("Backswing", f"{backswing_frames} frames", f"{backswing_time:.2f}s")
                
                with col2:
                    downswing_frames = impact - backswing_peak
                    downswing_time = downswing_frames / fps
                    st.metric("Downswing", f"{downswing_frames} frames", f"{downswing_time:.2f}s")
                
                with col3:
                    total_swing = impact - address
                    total_time = total_swing / fps
                    st.metric("Total Swing", f"{total_swing} frames", f"{total_time:.2f}s")
                
                with col4:
                    tempo_ratio = backswing_frames / downswing_frames if downswing_frames > 0 else 0
                    st.metric("Tempo Ratio", f"{tempo_ratio:.1f}:1", 
                             help="Backswing to downswing ratio (3:1 is ideal)")
            
            else:
                # Fallback if phase data not available
                peak_idx = df["wrist_y_smooth"].idxmax()
                peak_value = df["wrist_y_smooth"].iloc[peak_idx]
                ax.axvline(peak_idx, color='green', linestyle='--', alpha=0.7)
                ax.scatter(peak_idx, peak_value, color='green', s=100, zorder=5)
            
            ax.set_xlabel("Frame Number", fontsize=13, fontweight='bold')
            ax.set_ylabel("Wrist Height (normalized)", fontsize=13, fontweight='bold')
            ax.set_title("Complete Swing Analysis - All Phases", fontsize=15, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Make plot more readable
            plt.tight_layout()
            
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
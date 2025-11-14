import matplotlib.pyplot as plt
import streamlit as st

def plot_swing(my_wrist, pro_wrist=None, smoothed_my=None, smoothed_pro=None):
    # Create figure and axis properly
    fig, ax = plt.subplots(figsize=(12, 6))

    # Player swing
    ax.plot(my_wrist, label="Your Swing (Original)", alpha=0.6, color="blue")
    if smoothed_my is not None:
        ax.plot(smoothed_my, label="Your Swing (Smoothed)", linewidth=2, color="darkblue")

    # Pro swing
    if pro_wrist is not None:
        ax.plot(pro_wrist, label="Pro Swing (Original)", alpha=0.6, color="red")
    if smoothed_pro is not None:
        ax.plot(smoothed_pro, label="Pro Swing (Smoothed)", linewidth=2, color="darkred")

    # Labels, title, and legend
    ax.set_xlabel("Frame")
    ax.set_ylabel("Wrist Y Position (pixels)")
    ax.set_title("Swing Comparison")
    ax.legend()

    # Display in Streamlit
    st.pyplot(fig)


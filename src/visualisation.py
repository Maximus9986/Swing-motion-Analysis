import matplotlib.pyplot as plt

def plot_swing(wrist_y, smoothed=None):
    plt.figure(figsize=(12, 6))
    plt.plot(wrist_y, label="Original Wrist Y", alpha=0.6)
    if smoothed is not None:
        plt.plot(smoothed, label="Smoothed Wrist Y", linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("Wrist Y Position (pixels)")
    plt.title("Wrist Y Trajectory During Swing")
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt

def plot_swing(my_wrist, pro_wrist=None, smoothed_my=None, smoothed_pro=None):

    plt.figure(figsize=(12,6))

    # Player swing
    plt.plot(my_wrist, label="Your Swing (Original)", alpha=0.6, color="blue")
    if smoothed_my is not None:
        plt.plot(smoothed_my, label="Your Swing (Smoothed)", linewidth=2, color="darkblue")

    # Pro swing
    if pro_wrist is not None:
        plt.plot(pro_wrist, label="Pro Swing (Original)", alpha=0.6, color="red")
    if smoothed_pro is not None:
        plt.plot(smoothed_pro, label="Pro Swing (Smoothed)", linewidth=2, color="darkred")

    plt.xlabel("Frame")
    plt.ylabel("Wrist Y Position")
    plt.title("Swing Comparison")
    plt.legend()
    plt.show()

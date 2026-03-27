import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load CSVs
# -----------------------------
mp = pd.read_csv("src/mediapipe_yolo_predictions.csv")
cm = pd.read_csv("src/comotion_predictions.csv")
sn = pd.read_csv("src/swingnet_predictions.csv")

# -----------------------------
# Merge on ID
# -----------------------------
df = mp.merge(cm, on="id").merge(sn, on="id")

# -----------------------------
# Compute errors
# -----------------------------
df["diff_mp"] = df["mediapipe_impact"] - df["swingnet_impact"]
df["diff_cm"] = df["comotion_impact"] - df["swingnet_impact"]

df["err_mp"] = np.abs(df["diff_mp"])
df["err_cm"] = np.abs(df["diff_cm"])

# -----------------------------
# Plot histogram
# -----------------------------
plt.figure(figsize=(8,5))

bins = np.arange(0, 200, 5)

plt.hist(df["err_mp"], bins=bins, alpha=0.7, label="MediaPipe+YOLO")
plt.hist(df["err_cm"], bins=bins, alpha=0.5, label="CoMotion")

plt.xlabel("Absolute Error (frames)")
plt.ylabel("Number of Videos")
plt.title("Distribution of Impact Frame Errors (38 GolfDB Clips)")
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# Print summary (optional)
# -----------------------------
print("MediaPipe MAE:", df["err_mp"].mean())
print("CoMotion MAE:", df["err_cm"].mean())

print("MP within ±3:", (df["err_mp"] <= 3).sum())
print("CM within ±3:", (df["err_cm"] <= 3).sum())
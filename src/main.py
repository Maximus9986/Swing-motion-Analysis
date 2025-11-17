# from pose_tracking import extract_pose
# from swing_analysis import analyze_swing
# from visualisation import plot_swing
# import pandas as pd

# # --- Input videos ---
# player_swing = "../data/sample_swing1.mp4"
# player_output = "../data/sample_swing1_overlay.mp4"
# pro_swing = "../data/pro_swing.mp4"
# pro_output = "../data/pro_swing_overlay.mp4"

# # --- Extract pose + overlay video ---
# player_data = extract_pose(player_swing, output_path=player_output)
# df_my = pd.DataFrame(player_data)
# df_my = analyze_swing(df_my)
# df_my.to_csv("../data/player_swing_data.csv", index=False)

# pro_data = extract_pose(pro_swing, output_path=pro_output)
# df_pro = pd.DataFrame(pro_data)
# df_pro = analyze_swing(df_pro)
# df_pro.to_csv("../data/pro_swing_data.csv", index=False)

# # --- Plot comparison ---
# plot_swing(
#     my_wrist=df_my["y"],
#     pro_wrist=df_pro["y"],
#     smoothed_my=df_my["wrist_y_smooth"],
#     smoothed_pro=df_pro["wrist_y_smooth"]
# )

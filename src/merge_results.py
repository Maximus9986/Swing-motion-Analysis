import pandas as pd

mediapipe_df = pd.read_csv(r'C:\Users\User\FYP\Swing-motion-Analysis\src\mediapipe_yolo_predictions.csv')
swingnet_df  = pd.read_csv(r'C:\Users\User\FYP\Swing-motion-Analysis\src\swingnet_predictions.csv')

df = mediapipe_df[['id', 'mediapipe_impact']].merge(
    swingnet_df[['id', 'swingnet_impact']], on='id')

df['diff']     = df['mediapipe_impact'] - df['swingnet_impact']
df['abs_diff'] = df['diff'].abs()

# Per-video table
print("PER-VIDEO IMPACT COMPARISON (MediaPipe vs SwingNet):")
print(f"{'id':<8} {'swingnet':>10} {'mediapipe':>10} {'diff':>8}")
print("-" * 40)
for _, r in df.iterrows():
    print(f"{int(r['id']):<8} {int(r['swingnet_impact']):>10} {int(r['mediapipe_impact']):>10} {int(r['diff']):>+8}")

# Summary
print("\n" + "="*40)
print("SUMMARY")
print("="*40)
print(f"  Videos compared : {len(df)}")
print(f"  MAE             : {df['abs_diff'].mean():.2f} frames")
print(f"  Std dev         : {df['abs_diff'].std():.2f} frames")
print(f"  Mean error      : {df['diff'].mean():+.2f} frames")
print(f"  Within +-3      : {(df['abs_diff']<=3).mean()*100:.1f}%")
print(f"  Within +-5      : {(df['abs_diff']<=5).mean()*100:.1f}%")
print(f"  Within +-10     : {(df['abs_diff']<=10).mean()*100:.1f}%")

df.to_csv(r'C:\Users\User\FYP\Swing-motion-Analysis\src\impact_comparison.csv', index=False)
print(f"\nSaved to impact_comparison.csv")
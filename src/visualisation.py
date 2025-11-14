"""
Visualization Module
Creates plots and visual analysis for golf swing data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional
import config


def plot_swing_analysis(
    df: pd.DataFrame,
    metrics: Dict,
    return_fig: bool = True
) -> Optional[plt.Figure]:
    """
    Create comprehensive swing analysis visualization.
    
    Args:
        df: DataFrame with analyzed swing data
        metrics: Dictionary of swing metrics
        return_fig: If True, return figure object; if False, display plot
        
    Returns:
        Figure object if return_fig=True, None otherwise
    """
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Wrist trajectory over time
    ax1 = fig.add_subplot(gs[0, :])
    _plot_wrist_trajectory(ax1, df, metrics)
    
    # 2. Swing path (top view)
    ax2 = fig.add_subplot(gs[1, 0])
    _plot_swing_path_topview(ax2, df, metrics)
    
    # 3. Speed profile
    ax3 = fig.add_subplot(gs[1, 1])
    _plot_speed_profile(ax3, df, metrics)
    
    # 4. Phase distribution
    ax4 = fig.add_subplot(gs[2, 0])
    _plot_phase_distribution(ax4, metrics)
    
    # 5. 3D trajectory
    ax5 = fig.add_subplot(gs[2, 1], projection='3d')
    _plot_3d_trajectory(ax5, df, metrics)
    
    plt.suptitle("Golf Swing Analysis", fontsize=16, fontweight='bold')
    
    if return_fig:
        return fig
    else:
        plt.show()
        return None


def _plot_wrist_trajectory(ax, df: pd.DataFrame, metrics: Dict):
    """Plot wrist height over time with phase markers"""
    
    frames = df["frame"].values
    wrist_y = df["wrist_y_smooth"].values
    
    # Plot trajectory
    ax.plot(frames, wrist_y, label="Wrist Height", 
            color=config.GRAPH_COLORS["player_smooth"], linewidth=2)
    
    # Mark key phases
    phases = {
        "Address": metrics.get("address"),
        "Top of Backswing": metrics.get("backswing_peak"),
        "Impact": metrics.get("impact"),
        "Follow Through": metrics.get("follow_through")
    }
    
    colors = {
        "Address": "green",
        "Top of Backswing": "orange",
        "Impact": "red",
        "Follow Through": "purple"
    }
    
    for phase_name, frame_idx in phases.items():
        if frame_idx is not None and frame_idx < len(df):
            ax.axvline(
                frame_idx,
                color=colors.get(phase_name, "gray"),
                linestyle='--',
                alpha=0.7,
                label=phase_name
            )
    
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Wrist Height (normalized)", fontsize=11)
    ax.set_title("Wrist Trajectory During Swing", fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_swing_path_topview(ax, df: pd.DataFrame, metrics: Dict):
    """Plot swing path from top-down view (bird's eye)"""
    
    downswing_start = metrics.get("downswing_start", 0)
    impact = metrics.get("impact", len(df) - 1)
    
    # Get downswing path
    downswing = df.iloc[downswing_start:impact]
    
    x_path = downswing["wrist_x_smooth"].values
    z_path = downswing["wrist_z_smooth"].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x_path) | np.isnan(z_path))
    x_path = x_path[valid_mask]
    z_path = z_path[valid_mask]
    
    # Plot path with color gradient (time progression)
    if len(x_path) > 0:
        points = np.array([z_path, x_path]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        
        # Create color map from start (blue) to end (red)
        colors_array = plt.cm.RdYlBu_r(np.linspace(0, 1, len(segments)))
        
        lc = LineCollection(segments, colors=colors_array, linewidths=3)
        ax.add_collection(lc)
        
        # Mark start and end
        ax.scatter(z_path[0], x_path[0], s=100, c='green', 
                  marker='o', label='Start', zorder=5, edgecolors='black')
        ax.scatter(z_path[-1], x_path[-1], s=100, c='red', 
                  marker='*', label='Impact', zorder=5, edgecolors='black')
        
        # Draw target line (straight down)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Target Line')
        
        ax.set_xlabel("Forward/Back (z)", fontsize=11)
        ax.set_ylabel("Left/Right (x)", fontsize=11)
        
        # Add swing path info
        path_label = metrics.get("swing_path_label", "N/A")
        path_angle = metrics.get("swing_path_angle", 0)
        
        title = f"Swing Path (Top View)\n{path_label} | {path_angle:.1f}°"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.text(0.5, 0.5, "Insufficient data for path plot", 
               ha='center', va='center', transform=ax.transAxes)


def _plot_speed_profile(ax, df: pd.DataFrame, metrics: Dict):
    """Plot wrist speed profile over time"""
    
    frames = df["frame"].values
    speed = df["wrist_speed"].fillna(0).values
    
    # Plot speed
    ax.plot(frames, speed, label="Wrist Speed", 
            color='darkblue', linewidth=2)
    ax.fill_between(frames, speed, alpha=0.3, color='lightblue')
    
    # Mark impact
    impact_frame = metrics.get("impact")
    if impact_frame is not None and impact_frame < len(df):
        impact_speed = df.loc[impact_frame, "wrist_speed"]
        if not np.isnan(impact_speed):
            ax.scatter(impact_frame, impact_speed, s=200, c='red', 
                      marker='*', label=f'Impact Speed', zorder=5, edgecolors='black')
            ax.text(impact_frame, impact_speed * 1.1, 
                   f'{impact_speed:.3f}', ha='center', fontsize=9)
    
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Speed (normalized)", fontsize=11)
    ax.set_title("Wrist Speed During Swing", fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_phase_distribution(ax, metrics: Dict):
    """Plot swing phase durations as bar chart"""
    
    phases = {
        "Backswing": metrics.get("backswing_frames", 0),
        "Downswing": metrics.get("downswing_frames", 0)
    }
    
    colors_list = ['#2E86AB', '#A23B72']
    
    bars = ax.bar(phases.keys(), phases.values(), color=colors_list, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add tempo ratio
    tempo_ratio = metrics.get("tempo_ratio", 0)
    tempo_text = f"Tempo Ratio: {tempo_ratio:.2f}:1"
    
    ax.text(0.5, 0.95, tempo_text, 
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10)
    
    ax.set_ylabel("Number of Frames", fontsize=11)
    ax.set_title("Swing Phase Distribution", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')


def _plot_3d_trajectory(ax, df: pd.DataFrame, metrics: Dict):
    """Plot 3D wrist trajectory"""
    
    x = df["wrist_x_smooth"].values
    y = df["wrist_y_smooth"].values
    z = df["wrist_z_smooth"].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    
    if len(x) > 0:
        # Color by time progression
        colors_array = plt.cm.viridis(np.linspace(0, 1, len(x)))
        
        # Plot trajectory
        ax.plot(x, z, y, linewidth=2, alpha=0.6, color='blue')
        ax.scatter(x, z, y, c=colors_array, s=20, alpha=0.8)
        
        # Mark start and impact
        ax.scatter(x[0], z[0], y[0], s=100, c='green', 
                  marker='o', label='Start', edgecolors='black')
        
        impact_idx = metrics.get("impact", len(df) - 1)
        if impact_idx < len(x):
            ax.scatter(x[impact_idx], z[impact_idx], y[impact_idx], 
                      s=150, c='red', marker='*', label='Impact', edgecolors='black')
        
        ax.set_xlabel("Left/Right (x)", fontsize=9)
        ax.set_ylabel("Forward/Back (z)", fontsize=9)
        ax.set_zlabel("Height (y)", fontsize=9)
        ax.set_title("3D Wrist Trajectory", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
    else:
        ax.text2D(0.5, 0.5, "Insufficient data for 3D plot", 
                 ha='center', va='center', transform=ax.transAxes)


def plot_comparison(
    player_df: pd.DataFrame,
    pro_df: pd.DataFrame,
    player_metrics: Dict,
    pro_metrics: Dict,
    return_fig: bool = True
) -> Optional[plt.Figure]:
    """
    Compare player swing with professional swing.
    
    Args:
        player_df: Player's swing data
        pro_df: Professional's swing data
        player_metrics: Player's metrics
        pro_metrics: Professional's metrics
        return_fig: If True, return figure; if False, display
        
    Returns:
        Figure object if return_fig=True
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Wrist trajectory comparison
    ax1 = axes[0, 0]
    ax1.plot(player_df["wrist_y_smooth"], 
            label="Your Swing", 
            color=config.GRAPH_COLORS["player_smooth"],
            linewidth=2)
    ax1.plot(pro_df["wrist_y_smooth"], 
            label="Pro Swing", 
            color=config.GRAPH_COLORS["pro_smooth"],
            linewidth=2, linestyle='--')
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Wrist Height")
    ax1.set_title("Wrist Trajectory Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Speed comparison
    ax2 = axes[0, 1]
    ax2.plot(player_df["wrist_speed"].fillna(0), 
            label="Your Speed", 
            color=config.GRAPH_COLORS["player_smooth"],
            linewidth=2)
    ax2.plot(pro_df["wrist_speed"].fillna(0), 
            label="Pro Speed", 
            color=config.GRAPH_COLORS["pro_smooth"],
            linewidth=2, linestyle='--')
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Wrist Speed")
    ax2.set_title("Speed Profile Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Tempo comparison
    ax3 = axes[1, 0]
    tempos = {
        "Your Swing": player_metrics.get("tempo_ratio", 0),
        "Pro Swing": pro_metrics.get("tempo_ratio", 0),
        "Ideal": config.IDEAL_TEMPO_RATIO
    }
    colors_list = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax3.bar(tempos.keys(), tempos.values(), color=colors_list, alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom')
    ax3.set_ylabel("Tempo Ratio")
    ax3.set_title("Swing Tempo Comparison")
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Metrics summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    comparison_data = [
        ["Metric", "Your Swing", "Pro Swing"],
        ["Swing Path", 
         player_metrics.get("swing_path_label", "N/A")[:20],
         pro_metrics.get("swing_path_label", "N/A")[:20]],
        ["Path Angle", 
         f"{player_metrics.get('swing_path_angle', 0):.1f}°",
         f"{pro_metrics.get('swing_path_angle', 0):.1f}°"],
        ["Tempo Ratio", 
         f"{player_metrics.get('tempo_ratio', 0):.2f}",
         f"{pro_metrics.get('tempo_ratio', 0):.2f}"],
        ["Plane Quality",
         player_metrics.get("path_quality", "N/A").capitalize(),
         pro_metrics.get("path_quality", "N/A").capitalize()]
    ]
    
    table = ax4.table(cellText=comparison_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title("Metrics Comparison", fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle("Swing Comparison Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
        return None


def create_metrics_card(metrics: Dict) -> str:
    """
    Create a formatted text card with key metrics.
    
    Args:
        metrics: Dictionary of swing metrics
        
    Returns:
        Formatted string for display
    """
    
    card = f"""
╔══════════════════════════════════════════╗
║        SWING ANALYSIS SUMMARY            ║
╚══════════════════════════════════════════╝

🎯 SWING PATH
   Type: {metrics.get('swing_path_label', 'N/A')}
   Angle: {metrics.get('swing_path_angle', 0):.2f}°
   Expected: {metrics.get('expected_ball_flight', 'N/A')}
   Quality: {metrics.get('path_quality', 'N/A').capitalize()}

⏱️  SWING TEMPO
   Ratio: {metrics.get('tempo_ratio', 0):.2f}:1
   Assessment: {metrics.get('tempo_assessment', 'N/A')}
   
📐 SWING PLANE
   Consistency: {metrics.get('plane_consistency', 0)*100:.1f}%
   Assessment: {metrics.get('plane_assessment', 'N/A')}

📊 SWING PHASES
   Backswing: {metrics.get('backswing_duration_sec', 0):.2f}s
   Downswing: {metrics.get('downswing_duration_sec', 0):.2f}s
   Total: {metrics.get('total_swing_duration_sec', 0):.2f}s
    """
    
    return card
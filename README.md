# 🏌️ Golf Swing Motion Analysis System

A golf swing analysis application that uses computer vision and biomechanical heuristics to analyse golf swings from ordinary video.
The system focuses on robust swing phase detection, tempo, and body mechanics (early extension) using 2D pose estimation, making it accessible to players without expensive launch monitors or motion-capture systems.

## 🎯 Features

- **Pose Tracking**: Uses MediaPipe to extract 33-point pose landmarks from video
- **Swing Phase Detection**: Automatically identifies address, backswing top, impact, and finish phases
- **Tempo Analysis**: Measures backswing-to-downswing ratio (ideal ~3:1)
- **Arm Extension Analysis**: Evaluates elbow angle at impact for proper extension
- **Speed Timing**: Measures when maximum wrist speed occurs relative to impact
- **Early Extension Detection**: Side-view proxy for hip drift/rise during the swing
- **Visual Overlays**: Creates annotated videos with pose landmarks
- **Phase Visualization**: Displays key swing positions with optional skeleton overlay
- **Comprehensive Charts**: Wrist timeline, tempo breakdown, speed profiles, and score analysis

## 📁 Project Structure

```
FYP/
├── app.py                  # Streamlit application
├── pose_tracking.py        # MediaPipe pose extraction
├── swing_analysis.py       # Swing phase + biomechanical analysis
├── visualisation.py        # Plots and charts
├── requirements.txt
├── README.md
└── Data/
    ├── overlay.mp4
    ├── player_swing_analysis.csv
    └── sample_videos/  

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Webcam or video file of golf swing

### Setup Steps

1. **Clone or download the repository**
```bash
   cd /path/to/FYP
```

2. **Create a virtual environment (recommended)**
```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Mac/Linux
   source venv/bin/activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Run the application**
```bash
   python -m streamlit run app.py
```

5. **Open your browser**
   - The app should automatically open at `http://localhost:8501`
   - If not, manually navigate to this address

## 📖 Usage Guide

### Basic Usage

1. **Upload Video**
   - Click the file uploader
   - Select an MP4 video of your golf swing
   - Best results with side-view (facing or down-the-line) angle

2. **Analyze**
   - Click "Analyze Swing" button
   - Wait for pose extraction (may take 1-2 minutes)
   - Review your results

3. **View Results**
   - Overall score and rating displayed prominently
   - Phase frames show key positions in your swing
   - Detailed metrics for tempo, arm extension, speed timing, and early extension
   - Interactive charts for deeper analysis

4. **Download Results**
   - Download analysis CSV with all metrics

### Video Recording Tips

For best analysis results:
- **Camera position**: Side view (90° to target line)
- **Distance**: Full body visible with some margin
- **Lighting**: Good, even lighting
- **Frame rate**: 30fps or higher
- **Resolution**: 720p minimum, 1080p preferred
- **Background**: Uncluttered, contrasting with golfer
- **Duration**: Include full swing from setup to follow-through


## 📊 Metrics Explained

### Swing Phases
| Phase | Description |
|-------|-------------|
| **Address** | Initial setup position (stable period before movement) |
| **Top of Backswing** | Highest point of wrist during backswing (minimum Y value, requires >0.2 drop) |
| **Impact** | Ball contact point (detected via wrist trajectory peak after downswing) |
| **Finish** | End of follow-through (when wrist speed stabilizes) |

### Tempo Ratio
- **Definition**: Backswing frames ÷ Downswing frames
- **Ideal**: ~3:1 (similar to tour professionals)
- **Interpretation**:
  - 2.0–3.5:1 = Good tempo
  - <2.0:1 = Quick backswing, consider slowing down
  - >3.5:1 = Slow downswing, consider accelerating through impact

### Arm Extension
- **Metric**: Elbow angle at impact (degrees)
- **Ideal**: >155° (fully extended)
- **Ratings**:
  - >155° = Excellent (fully extended)
  - 140–155° = Good (slight bend is normal)
  - 125–140° = Moderate
  - <125° = Needs improvement (chicken wing)

### Speed Timing
- **Metric**: Frame difference between max wrist speed and impact
- **Ideal**: Max speed at or just before impact (±2 frames)
- **Ratings**:
  - ±2 frames = Excellent
  - ±4 frames = Good
  - ±6 frames = Moderate
  - >6 frames = Early/Late release

### Early Extension
- **Definition**: Hip movement toward the ball during downswing
- **Metric**: Normalized hip drift (X) and rise (Y) from address to impact
- **Score**: 0–100 (lower is better)
- **Ratings**:
  - 0–30 = Low risk (good)
  - 31–60 = Moderate risk
  - 61–100 = High risk

### Overall Score
Weighted combination of:
- Tempo (25 points)
- Arm Extension (25 points)
- Speed Timing (25 points)
- Early Extension (25 points)

| Score | Rating |
|-------|--------|
| 85–100 | Excellent |
| 70–84 | Good |
| 55–69 | Average |
| <55 | Needs Work |

## 🔬 Technical Details

### Pose Detection
- Uses Google MediaPipe Pose
- Extracts 33 landmarks per frame
- Key joints tracked: wrists, elbows, shoulders, hips
- Smoothing applied via Savitzky-Golay filter (window=7, polyorder=2)

### Phase Detection Algorithm

1. **Address Detection**
   - Finds stable segment early in video using rolling standard deviation
   - Identifies period of minimal wrist movement

2. **Backswing Start**
   - Detects first consistent downward movement from address
   - Requires significant future drop (≥30% of total range or 0.2 minimum)

3. **Top of Backswing**
   - Finds troughs (local minima) in wrist Y trajectory
   - **Requires minimum 0.2 drop from backswing start** to avoid jitter detection
   - Validates with movement after the trough

4. **Impact Detection**
   - Finds first peak in wrist Y after top of backswing
   - Represents the point where wrist rises back up through the ball

5. **Finish Detection**
   - Monitors wrist speed after impact
   - Finish when speed drops below 12% of peak for 12+ consecutive frames

### Data Processing
- Savitzky-Golay smoothing for noise reduction
- Forward/backward fill for missing data
- Normalized metrics using torso length as reference

## 🐛 Troubleshooting

### Common Issues

**"No pose detected" or low detection rate**
- Ensure full body is visible in frame
- Improve lighting
- Reduce background clutter
- Try different camera angle

**"Video too short" error**
- Video must be at least 30 frames (1 second at 30fps)
- Include complete swing from setup to follow-through

**Inaccurate swing path**
- Ensure side view (not face-on)
- Check that golfer doesn't step out of frame
- Verify good pose detection rate (>80%)

**Slow processing**
- Processing time ~1-2 minutes for typical video
- Longer videos take more time
- Close other applications to free resources

## 📚 Dependencies

- **streamlit**: Web application framework
- **mediapipe**: Pose estimation
- **opencv-python**: Video processing
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib**: Visualization
- **scipy**: Statistical analysis

## 🎓 Academic Context

This is a final year project for Computer Science at the University of Birmingham. The system demonstrates:
- Computer vision applications in sports analysis
- Real-time pose estimation techniques
- Biomechanical analysis algorithms
- Signal processing for motion analysis
- Interactive data visualization
- Full-stack application development

## 🔮 Future Enhancements

Potential improvements for future versions:
- [ ] Club detection using GolfPose (HRNet + YOLOX) for more accurate impact detection
- [ ] 3D pose estimation integration (CoMotion/SMPL)
- [ ] Real-time webcam analysis
- [ ] Multiple swing comparison
- [ ] Hip/shoulder rotation analysis
- [ ] Pro swing comparison using DTW
- [ ] Mobile app version
- [ ] Historical tracking database
- [ ] PDF report generation
## 📝 Code Quality Improvements

## 📄 License

This project is for academic purposes. Please cite if using for research or educational purposes.

## 👤 Author

**Lim Maximus**
- University of Birmingham
- Computer Science Final Year Project
- Year: 2025-2026

## 🙏 Acknowledgments

- MediaPipe team at Google for pose estimation
- Streamlit for the web framework
- Golf instructors and biomechanics research

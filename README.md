# 🏌️ Golf Swing Motion Analysis System

A golf swing analysis application that uses computer vision and biomechanical heuristics to analyse golf swings from ordinary video.
The system combines MediaPipe 2D pose estimation with YOLOv8 club detection for robust swing phase detection, tempo analysis, and body mechanics evaluation, making it accessible to players without expensive launch monitors or motion-capture systems.

## 🎯 Features

- **Pose Tracking**: MediaPipe extracts pose landmarks from video, tracking wrist, elbow, shoulder, and hip joints
- **Club Detection**: YOLOv8 custom-trained model detects the clubhead per frame, refining impact detection beyond wrist trajectory alone
- **Handedness Support**: Supports both right-handed and left-handed golfers via a sidebar toggle, automatically tracking the correct lead arm
- **Swing Phase Detection**: Automatically identifies address, backswing top, impact, and finish phases
- **Tempo Analysis**: Measures backswing-to-downswing ratio (ideal ~3:1)
- **Arm Extension Analysis**: Evaluates elbow angle at impact for proper extension
- **Speed Timing**: Measures when maximum wrist speed occurs relative to impact
- **Early Extension Detection**: Side-view proxy measuring normalised hip drift and rise during the swing
- **Phase Visualisation**: Displays key swing positions with optional skeleton overlay
- **Comprehensive Charts**: Wrist timeline, tempo breakdown, speed profiles, and score analysis

## 📁 Project Structure

```
src/
├── app.py                  # Streamlit application
├── pose_tracking.py        # MediaPipe pose extraction
├── club_tracking.py        # YOLOv8 clubhead detection + impact refinement
├── swing_analysis.py       # Swing phase + biomechanical analysis
├── visualisation.py        # Plots and charts
├── models/
│   └── best.pt             # Custom-trained YOLOv8 club detection model
├── requirements.txt
└── README.md
Data/
├── player_swing_analysis.csv
└── sample_videos/
```

## 🚀 Installation

### Prerequisites
- Python 3.10 (Anaconda environment recommended)
- pip package manager
- Video file of a golf swing (side/down-the-line view)

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/Maximus9986/Swing-motion-Analysis.git
cd Swing-motion-Analysis
```

2. **Create a conda environment (recommended)**
```bash
conda create -n FYP python=3.10
conda activate FYP
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
cd src
python -m streamlit run app.py
```

5. **Open your browser**
   - The app should automatically open at `http://localhost:8501`
   - If not, manually navigate to this address

## 📖 Usage Guide

### Basic Usage

1. **Select Handedness**
   - Use the sidebar radio button to select the golfer's dominant hand (right or left)
   - This determines which arm is tracked as the lead arm

2. **Upload Video**
   - Click the file uploader
   - Select an MP4, MOV, or AVI video of your golf swing
   - Best results with a down-the-line camera angle

3. **View Results**
   - Analysis runs automatically on upload
   - Overall score and rating displayed prominently
   - Phase frames show key positions in your swing
   - Detailed metrics for tempo, arm extension, speed timing, and early extension
   - Interactive charts across four tabs for deeper analysis

4. **Download Results**
   - Download the analysis CSV with all metrics using the button at the bottom

### Video Recording Tips

For best analysis results:
- **Camera position**: Down-the-line (behind the golfer, along the target line)
- **Distance**: Full body visible with some margin
- **Lighting**: Good, even lighting
- **Frame rate**: 30fps or higher
- **Resolution**: 720p minimum, 1080p preferred
- **Background**: Uncluttered, contrasting with golfer
- **Duration**: Include a short address setup before swinging through to follow-through

## 📊 Metrics Explained

### Swing Phases
| Phase | Description |
|-------|-------------|
| **Address** | Initial setup position (stable period before movement begins) |
| **Top of Backswing** | Deepest trough in wrist Y trajectory (requires minimum 0.35 drop from backswing start to filter jitter) |
| **Impact** | Ball contact point — wrist trajectory provides initial estimate, then YOLOv8 clubhead Y refines within an asymmetric window (5 frames back, 12 frames forward) to find the lowest clubhead position |
| **Finish** | End of follow-through (detected when wrist Y stabilises within a 10-frame window, with relaxed threshold and minimum-speed fallbacks) |

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
  - <125° = Needs improvement

### Speed Timing
- **Metric**: Frame difference between max wrist speed and impact
- **Ideal**: Max speed at or just before impact (within 2 frames)
- **Ratings**:
  - Within 2 frames = Excellent
  - Within 4 frames = Good
  - Within 6 frames = Moderate
  - Beyond 6 frames = Needs work

### Early Extension
- **Definition**: Hip movement toward the ball during the downswing
- **Metric**: Normalised hip drift (X) and rise (Y) from address to impact, using torso length as the reference scale
- **Score**: 0–100 (lower is better)
- **Ratings**:
  - 0–30 = Low risk (good)
  - 31–60 = Moderate risk
  - 61–100 = High risk

### Overall Score
Equal-weighted combination of four components (25 points each):
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
- Uses Google MediaPipe Pose with model complexity 2 (most accurate)
- Tracks lead-arm joints: wrist, elbow, shoulder, and hip
- Lead arm is automatically selected based on handedness (left side for right-handed golfers, right side for left-handed golfers)
- Missing landmarks are interpolated linearly across frames
- All joint trajectories smoothed via Savitzky-Golay filter (window=7, polyorder=2)

### Club Detection
- Custom-trained YOLOv8 pose model detects the clubhead in each frame
- Uses keypoint Y coordinates when available, falls back to bounding box bottom edge
- Smoothed with the same Savitzky-Golay filter as pose data
- Provides per-frame `clubhead_y`, `clubhead_valid`, and `clubhead_y_smooth` values

### Phase Detection Algorithm
Phase detection follows a top-down strategy: the backswing top is found first, then the address is located by walking backwards from that anchor point.

1. **Top of Backswing (found first)**
   - Searches the first 500 frames of smoothed wrist Y for the major trough
   - A baseline wrist height is established from the median of the first quarter of the search window
   - Candidate troughs detected using scipy.signal.find_peaks on the inverted signal, accepted only if the wrist has descended by at least 0.35 units below the baseline
   - Each candidate validated by checking that significant movement follows within a 25-frame lookahead
   - Falls back to the deepest valid trough, then the absolute minimum if needed

2. **Address and Backswing Start (found by walking backwards from top)**
   - Starting from the backswing top, walks backwards frame by frame
   - At each frame, checks two conditions: (1) per-frame wrist Y drop rate below 0.008, and (2) preceding 8-frame window has wrist Y range below 0.04
   - The first frame satisfying both conditions is taken as the backswing start
   - Address is placed 5 frames before the backswing start, capturing a settled setup position
   - This top-down approach prevents videos with extended pre-swing footage (waggles, practice swings) from detecting address at frame 0

3. **Impact Detection (two-stage)**
   - **Stage 1 (wrist)**: Finds the first peak in wrist Y after the top of backswing, within a 120-frame search window
   - **Stage 2 (YOLO refinement)**: Searches an asymmetric window around the wrist estimate (5 frames back, 12 frames forward) for the frame with the lowest clubhead Y position. The forward bias accounts for the fact that the wrist peak typically fires a few frames before the club reaches the ball.
   - Falls back to wrist-only if no valid YOLO detections exist in the window

4. **Finish Detection**
   - Searches for a 10-frame window where wrist Y range is below 0.05 (tight threshold)
   - If not found, retries with a relaxed threshold of 0.10
   - If still not found, falls back to the frame with minimum wrist speed after impact

### Handedness
- Selecting "left" or "right" changes which MediaPipe landmarks are extracted (right-side joints for left-handed golfers, left-side joints for right-handed golfers)
- All downstream analysis runs identically regardless of handedness — the swing biomechanics are mirrored but the calculations are the same

### Data Processing
- Savitzky-Golay smoothing for noise reduction on all joint and clubhead trajectories
- Linear interpolation for missing pose detections
- Forward/backward fill for remaining gaps
- Normalised metrics using torso length (shoulder-to-hip distance) as reference

## 🐛 Troubleshooting

### Common Issues

**"No pose detected" or low detection rate**
- Ensure full body is visible in frame
- Improve lighting
- Reduce background clutter
- Try a different camera angle

**"Video too short" error**
- Video must be at least 30 frames (1 second at 30fps)
- Include complete swing from setup to follow-through

**Inaccurate phase detection**
- Ensure camera is positioned down-the-line (beside the golfer)
- Check that the golfer doesn't step out of frame
- Verify good pose detection rate (>80%)
- Ensure the correct handedness is selected in the sidebar

**YOLO club tracking failed**
- The app will fall back to wrist-only impact detection
- Ensure the club is visible in the video
- Good lighting and contrast help detection confidence

**Slow processing**
- Processing time is typically 1–2 minutes depending on video length
- YOLO inference is the most time-intensive step
- Results are cached per video — reanalysis is instant unless the video or handedness changes

## 📚 Dependencies

- **streamlit**: Web application framework
- **mediapipe**: Pose estimation
- **ultralytics**: YOLOv8 club detection
- **opencv-python**: Video processing
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib**: Visualisation
- **scipy**: Signal processing (Savitzky-Golay smoothing, peak detection)
- **Pillow**: Image handling for phase frame display

## 🔮 Future Enhancements

Potential improvements for future versions:
- [ ] 3D pose estimation integration 
- [ ] Real-time webcam analysis
- [ ] Multiple swing comparison
- [ ] Hip/shoulder rotation analysis
- [ ] Pro swing comparison using Dynamic Time Warping
- [ ] Mobile app version
- [ ] Historical tracking database
- [ ] PDF report generation

## 👤 Author

**Lim Maximus**
- University of Birmingham
- Computer Science Final Year Project
- Year: 2025–2026

## 🙏 Acknowledgments

- MediaPipe team at Google for pose estimation
- Ultralytics for YOLOv8
- Streamlit for the web framework
- Golf instructors and biomechanics research
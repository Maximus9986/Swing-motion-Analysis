# 🏌️ Golf Swing Motion Analysis System

A comprehensive golf swing analysis application using computer vision and biomechanical analysis. This system extracts pose data from swing videos and provides detailed metrics on swing path, tempo, plane consistency, and more.

## 🎯 Features

- **Pose Tracking**: Uses MediaPipe to extract 33-point pose landmarks from video
- **Swing Phase Detection**: Automatically identifies setup, backswing, downswing, impact, and follow-through
- **Swing Path Analysis**: Calculates swing path angle and classifies (hook, draw, straight, fade, slice)
- **Tempo Analysis**: Measures backswing-to-downswing ratio (ideal 3:1)
- **Swing Plane Consistency**: Evaluates how consistently you stay on plane
- **Visual Overlays**: Creates annotated videos with pose landmarks
- **Comprehensive Charts**: Multiple visualizations including 3D trajectory, speed profiles, and more

## 📁 Project Structure

```
FYP/
├── app.py                  # Main Streamlit application
├── config.py              # Configuration and constants
├── pose_tracking.py       # Pose extraction module
├── swing_analysis.py      # Swing analysis algorithms
├── visualisation.py       # Plotting and visualization
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── Data/                 # Data folder
    ├── player_swing_data.csv
    ├── pro_swing.mp4
    ├── sample_swing.mp4
    ├── sample_swing1.mp4
    └── sample_swing2.mp4
```

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
   streamlit run app.py
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

3. **Download Results**
   - Download the pose overlay video
   - Download analysis charts
   - Download raw CSV data

### Video Recording Tips

For best analysis results:
- **Camera position**: Side view (90° to target line)
- **Distance**: Full body visible with some margin
- **Lighting**: Good, even lighting
- **Frame rate**: 30fps or higher
- **Resolution**: 720p minimum, 1080p preferred
- **Background**: Uncluttered, contrasting with golfer
- **Duration**: Include full swing from setup to follow-through

## 🔧 Configuration

Edit `config.py` to customize analysis parameters:

```python
# Smoothing
SMOOTHING_WINDOW = 5  # Frames for moving average

# Swing path thresholds (degrees)
SWING_PATH_THRESHOLDS = {
    "strong_in_to_out": 8,
    "moderate_in_to_out": 3,
    "neutral_upper": 3,
    "neutral_lower": -3,
    "moderate_out_to_in": -8,
}

# Tempo
IDEAL_TEMPO_RATIO = 3.0  # Ideal backswing:downswing
```

## 📊 Metrics Explained

### Swing Path Analysis
- **Path Angle**: Direction of club path relative to target line
  - Positive = In-to-out (draws/hooks)
  - Negative = Out-to-in (fades/slices)
  - Near 0 = Straight
- **Path Quality**: How linear the path is (R² score)
  - Excellent: >0.85
  - Good: 0.70-0.85
  - Fair: 0.50-0.70
  - Poor: <0.50

### Swing Tempo
- **Tempo Ratio**: Backswing frames / Downswing frames
  - Ideal: 3:1 (similar to tour pros)
  - Higher: Slower, smoother backswing
  - Lower: Quicker, more aggressive backswing

### Swing Plane
- **Consistency Score**: How well you maintain swing plane (0-100%)
  - >80%: Excellent
  - 60-80%: Good
  - 40-60%: Fair
  - <40%: Inconsistent

### Swing Phases
- **Setup**: Initial address position
- **Backswing**: From address to top of swing
- **Downswing**: From top to impact
- **Impact**: Ball contact point
- **Follow-through**: After impact to finish

## 🔬 Technical Details

### Pose Detection
- Uses Google MediaPipe Pose
- Extracts 33 3D landmarks per frame
- Tracks: wrists, elbows, shoulders, hips, knees, ankles
- Confidence thresholds ensure quality

### Swing Path Algorithm
1. Detect swing phases using wrist trajectory
2. Extract downswing segment
3. Perform linear regression on wrist path (X-Z plane)
4. Calculate angle from regression slope
5. Classify based on configurable thresholds

### Phase Detection
1. **Backswing peak**: Maximum wrist height
2. **Address**: Most stable position before backswing
3. **Impact**: Maximum speed during downswing
4. **Follow-through**: Motion deceleration after impact

### Data Smoothing
- Rolling average with configurable window (default: 5 frames)
- Linear interpolation for missing data
- Preserves key features while reducing noise

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
- Computer vision applications
- Real-time pose estimation
- Biomechanical analysis
- Data visualization
- Software engineering best practices

## 🔮 Future Enhancements

Potential improvements for future versions:
- [ ] Real-time webcam analysis
- [ ] Multiple golfer comparison
- [ ] Club face angle estimation
- [ ] Hip/shoulder rotation analysis
- [ ] DTW-based pro comparison
- [ ] Machine learning swing classification
- [ ] Mobile app version
- [ ] Database for historical tracking
- [ ] Slow-motion playback with annotations
- [ ] Export to PDF reports

## 📝 Code Quality Improvements

This improved version includes:

### From Original Code
- ✅ Fixed hardcoded values
- ✅ Added proper error handling
- ✅ Improved swing path calculation
- ✅ Better phase detection
- ✅ Used overlay video in UI
- ✅ Added comprehensive documentation

### New Features
- ✅ Configuration file for all constants
- ✅ Type hints throughout
- ✅ Comprehensive error messages
- ✅ Data validation
- ✅ Linear interpolation for missing data
- ✅ Speed and acceleration metrics
- ✅ 3D trajectory visualization
- ✅ Improved UI with tabs and metrics
- ✅ Download capabilities
- ✅ Progress indicators
- ✅ Better color schemes
- ✅ Professional formatting

## 📄 License

This project is for academic purposes. Please cite if using for research or educational purposes.

## 👤 Author

**Maximus**
- University of Birmingham
- Computer Science Final Year Project
- Year: 2025-2026

## 🙏 Acknowledgments

- MediaPipe team at Google for pose estimation
- Streamlit for the web framework
- Golf instructors and biomechanics research

## 📧 Contact

For questions or issues, please create an issue in the repository or contact through the university.

---

**Note**: This system provides analysis for educational and training purposes. For professional golf instruction, please consult a certified golf professional.
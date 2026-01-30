import sys
import os

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)

print(f"\nPython executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Check if in conda environment
if 'CONDA_DEFAULT_ENV' in os.environ:
    print(f"Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")
else:
    print("⚠️  Not in a conda environment!")

# Try importing packages
packages = {
    'mediapipe': 'MediaPipe',
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'scipy': 'SciPy',
    'streamlit': 'Streamlit'
}

print("\n" + "=" * 60)
print("PACKAGE CHECK")
print("=" * 60)

for pkg, name in packages.items():
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {name:15s} {version}")
    except ImportError:
        print(f"❌ {name:15s} NOT INSTALLED")

print("=" * 60)
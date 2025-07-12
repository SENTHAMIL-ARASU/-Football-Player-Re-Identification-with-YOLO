# Player Re-Identification System - Setup Guide

## 🚀 Quick Start Setup

### Prerequisites Check

Before starting, ensure your system meets these requirements:

- **Operating System**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: At least 2GB free space
- **GPU**: Optional but recommended for faster processing

### Step-by-Step Installation

#### 1. Environment Setup

```bash
# Create a virtual environment (recommended)
python -m venv player_reid_env

# Activate virtual environment
# Windows:
player_reid_env\Scripts\activate
# macOS/Linux:
source player_reid_env/bin/activate
```

#### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python test_imports.py
```

#### 3. Download Model and Video

Ensure these files are in your project directory:
- `best (2).pt` (186MB YOLO model)
- `15sec_input_720p.mp4` (4.9MB input video)

#### 4. Run the System

```bash
# Complete implementation (recommended)
python player_reid_complete.py
```

## 🔧 Detailed Configuration

### Environment Variables

Create a `.env` file for custom configuration:

```bash
# Model settings
MODEL_PATH=best (2).pt
CONFIDENCE_THRESHOLD=0.3
REID_DISTANCE=50

# Video settings
INPUT_VIDEO=15sec_input_720p.mp4
OUTPUT_VIDEO=output_complete.mp4

# Performance settings
USE_GPU=true
BATCH_SIZE=1
```

### Advanced Configuration

#### GPU Acceleration Setup

For NVIDIA GPUs:

```bash
# Install CUDA toolkit (if not already installed)
# Download from: https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Memory Optimization

For systems with limited RAM:

```python
# In player_reid_complete.py, modify these settings:
import gc

# Add garbage collection after each frame
gc.collect()

# Reduce batch size
batch_size = 1

# Use smaller input resolution
input_size = (256, 448)  # Reduced from (384, 640)
```

## 🐛 Troubleshooting Guide

### Common Installation Issues

#### Issue 1: Import Errors

**Error**: `ModuleNotFoundError: No module named 'ultralytics'`

**Solution**:
```bash
# Reinstall ultralytics
pip uninstall ultralytics
pip install ultralytics==8.0.196

# Or install from source
pip install git+https://github.com/ultralytics/ultralytics.git
```

#### Issue 2: OpenCV Installation Problems

**Error**: `ImportError: No module named 'cv2'`

**Solution**:
```bash
# Windows
pip install opencv-python==4.8.1.78

# macOS
brew install opencv
pip install opencv-python

# Linux
sudo apt-get install python3-opencv
pip install opencv-python
```

#### Issue 3: PyTorch Installation Issues

**Error**: `ImportError: No module named 'torch'`

**Solution**:
```bash
# CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU version (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Runtime Issues

#### Issue 1: Model Not Found

**Error**: `❌ Error: Model file 'best (2).pt' not found!`

**Solution**:
```bash
# Check if model file exists
ls -la "best (2).pt"

# Download model if missing
# Contact instructor for model file or download from provided link
```

#### Issue 2: Video Not Found

**Error**: `❌ Error: Input video '15sec_input_720p.mp4' not found!`

**Solution**:
```bash
# Check if video file exists
ls -la 15sec_input_720p.mp4

# Ensure video is in correct format
ffmpeg -i your_video.mp4 -c:v libx264 -crf 23 15sec_input_720p.mp4
```

#### Issue 3: Memory Errors

**Error**: `MemoryError` or `CUDA out of memory`

**Solution**:
```python
# Reduce batch size
batch_size = 1

# Use CPU instead of GPU
device = 'cpu'

# Reduce input resolution
input_size = (256, 448)

# Add memory cleanup
import gc
gc.collect()
```

#### Issue 4: Slow Processing

**Symptoms**: Processing takes more than 15 minutes

**Solutions**:
```python
# 1. Use GPU acceleration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Reduce input resolution
input_size = (256, 448)

# 3. Increase confidence threshold
conf_threshold = 0.5

# 4. Skip frames (process every nth frame)
frame_skip = 2
```

### Performance Optimization

#### GPU Optimization

```python
# Enable mixed precision for faster processing
from torch.cuda.amp import autocast

with autocast():
    detections = model(frame)[0]
```

#### Memory Optimization

```python
# Clear GPU cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use smaller data types
import numpy as np
frame = frame.astype(np.float32)
```

#### CPU Optimization

```python
# Use multiple CPU cores
import multiprocessing
num_workers = multiprocessing.cpu_count()

# Optimize OpenCV operations
cv2.setUseOptimized(True)
cv2.setNumThreads(num_workers)
```

## 🔍 Testing and Validation

### System Health Check

Run the comprehensive test suite:

```bash
# Test all dependencies
python test_imports.py

# Expected output:
# ✅ OpenCV version: 4.8.1.78
# ✅ NumPy version: 1.24.3
# ✅ Ultralytics YOLO imported successfully
# ✅ Math module imported successfully
# ✅ OS module imported successfully
# ✅ Model loaded successfully!
```

### Performance Benchmarking

```bash
# Run performance test
python -c "
import time
import cv2
from ultralytics import YOLO

# Load model
model = YOLO('best (2).pt')

# Load video
cap = cv2.VideoCapture('15sec_input_720p.mp4')
ret, frame = cap.read()

# Benchmark detection
start_time = time.time()
for _ in range(10):
    detections = model(frame)[0]
end_time = time.time()

print(f'Average detection time: {(end_time - start_time)/10:.3f} seconds')
cap.release()
"
```

### Output Validation

Check output video quality:

```bash
# Verify output video properties
ffprobe output_complete.mp4

# Expected properties:
# - Duration: 15 seconds
# - Resolution: 1280x720
# - Frame rate: 25 fps
# - Codec: H.264
```

## 🛠️ Customization Options

### Model Configuration

```python
# Custom model settings
model_config = {
    'conf': 0.3,        # Confidence threshold
    'iou': 0.5,         # NMS IoU threshold
    'max_det': 100,     # Maximum detections
    'device': 'auto'    # Device selection
}

model = YOLO('best (2).pt')
model.conf = model_config['conf']
model.iou = model_config['iou']
```

### Tracking Configuration

```python
# Custom tracking parameters
tracking_config = {
    'reid_distance': 50,    # Re-identification threshold
    'min_track_length': 5,  # Minimum track length
    'max_disappeared': 30,  # Max frames before track deletion
    'smooth_factor': 0.8    # Position smoothing factor
}
```

### Visualization Configuration

```python
# Custom visualization settings
viz_config = {
    'show_boxes': True,     # Show bounding boxes
    'show_ids': True,       # Show player IDs
    'show_centroids': True, # Show centroids
    'show_stats': True,     # Show statistics panel
    'colors': {
        'new_player': (0, 255, 0),      # Green
        'reidentified': (0, 255, 255),  # Yellow
        'background': (0, 0, 0)         # Black
    }
}
```

## 📊 Monitoring and Logging

### Enable Debug Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('player_reid.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Performance Monitoring

```python
import psutil
import time

def monitor_performance():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    print(f"CPU: {cpu_percent}%, Memory: {memory_percent}%, GPU: {gpu_memory/1024**2:.1f}MB")
```

## 🔒 Security Considerations

### File Permissions

```bash
# Set appropriate file permissions
chmod 644 *.py
chmod 644 *.md
chmod 644 requirements.txt
chmod 600 .env  # If using environment variables
```

### Model Security

```python
# Validate model file integrity
import hashlib

def verify_model(model_path):
    expected_hash = "your_model_hash_here"
    with open(model_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash == expected_hash
```

## 📞 Support and Resources

### Getting Help

1. **Check Documentation**: Review README.md and this setup guide
2. **Run Tests**: Execute `python test_imports.py` to verify installation
3. **Check Logs**: Review error messages and system logs
4. **Community Support**: Use GitHub issues for bug reports

### Useful Commands

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check OpenCV version
python -c "import cv2; print(cv2.__version__)"

# Monitor system resources
top  # Linux/macOS
tasklist  # Windows
```

### Additional Resources

- **Ultralytics Documentation**: https://docs.ultralytics.com/
- **OpenCV Documentation**: https://docs.opencv.org/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Project Repository**: [Your GitHub repository URL]

---

**Setup Guide Version**: 1.0  
**Last Updated**: December 2024  
**Compatibility**: Python 3.8+, Windows/macOS/Linux 
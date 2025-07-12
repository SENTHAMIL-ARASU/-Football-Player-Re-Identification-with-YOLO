# Player Re-Identification System

## 🎯 Project Overview

This project implements a real-time player re-identification system for sports video analysis. The system detects players in a 15-second football video and maintains consistent player IDs even when players leave and re-enter the frame.

## 🚀 Features

- **Real-time Player Detection**: Uses YOLO model to detect players, referees, and the ball
- **Player Re-identification**: Maintains consistent IDs when players re-enter the frame
- **Visual Tracking**: Displays bounding boxes and unique player IDs
- **Performance Metrics**: Real-time statistics and progress tracking
- **Error Handling**: Robust error handling and graceful degradation

## 📁 Project Structure

```
player-reid-liat/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── player_reid.py              # Basic implementation
├── player_reid_improved.py     # Enhanced version with error handling
├── player_reid_complete.py     # Complete implementation with all features
├── test_imports.py             # Dependency testing script
├── best (2).pt                 # YOLO model file (186MB)
├── 15sec_input_720p.mp4        # Input video (4.9MB)
├── output.mp4                  # Basic output
├── output_improved.mp4         # Enhanced output
├── output_complete.mp4         # Complete output with all features
└── docs/
    ├── REPORT.md               # Technical report
    └── SETUP.md                # Detailed setup instructions
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Windows 10/11 (tested on Windows 10.0.26100)
- At least 4GB RAM
- GPU recommended for faster processing

### Step 1: Clone or Download

```bash
# If using Git
git clone <repository-url>
cd player-reid-liat

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Test if all dependencies are working
python test_imports.py
```

You should see:
```
✅ OpenCV version: 4.x.x
✅ NumPy version: 1.x.x
✅ Ultralytics YOLO imported successfully
✅ Math module imported successfully
✅ OS module imported successfully
✅ Model loaded successfully!
```

## 🎬 Usage

### Quick Start

1. **Basic Implementation**:
   ```bash
   python player_reid.py
   ```

2. **Enhanced Version** (Recommended):
   ```bash
   python player_reid_improved.py
   ```

3. **Complete Version** (All Features):
   ```bash
   python player_reid_complete.py
   ```

### Input Requirements

- Place your input video as `15sec_input_720p.mp4` in the project directory
- The system expects 720p video at 25 FPS
- Video should be in MP4 format

### Output

- **Basic**: `output.mp4` - Simple player tracking
- **Enhanced**: `output_improved.mp4` - Better error handling and progress tracking
- **Complete**: `output_complete.mp4` - Full features with re-identification visualization

## 📊 Performance Metrics

| Version | Processing Time | Frames | Players Tracked | Re-identifications |
|---------|----------------|--------|-----------------|-------------------|
| Basic | ~10 minutes | 375/375 | 58 | Basic tracking |
| Enhanced | ~10.7 minutes | 375/375 | 58 | Improved tracking |
| Complete | ~9.2 minutes | 375/375 | 58 | 105 re-identifications |

## 🔧 Configuration

### Model Settings

- **Model File**: `best (2).pt` (186MB YOLO model)
- **Confidence Threshold**: 0.3 (30%)
- **Re-identification Distance**: 50 pixels
- **Input Resolution**: 384x640 (optimized for speed)

### Customization

You can modify these parameters in the code:

```python
# In player_reid_complete.py
conf_threshold = 0.3        # Detection confidence
reid_distance = 50          # Re-identification threshold
input_size = (384, 640)     # Processing resolution
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**:
   ```
   ❌ Error: Model file 'best (2).pt' not found!
   ```
   **Solution**: Ensure the model file is in the project directory

2. **Video not found**:
   ```
   ❌ Error: Input video '15sec_input_720p.mp4' not found!
   ```
   **Solution**: Place your input video in the project directory

3. **Import errors**:
   ```
   ❌ Import error: No module named 'ultralytics'
   ```
   **Solution**: Run `pip install -r requirements.txt`

4. **Slow processing**:
   - Use GPU if available
   - Reduce input video resolution
   - Close other applications to free up memory

### Performance Optimization

- **GPU Acceleration**: Install CUDA for faster processing
- **Memory**: Ensure at least 4GB free RAM
- **Storage**: Ensure sufficient disk space for output videos

## 📈 Expected Results

When running successfully, you should see:

```
📦 Loading model from best (2).pt...
✅ Model loaded successfully!
🎥 Loading video from 15sec_input_720p.mp4...
📐 Video dimensions: 1280x720, FPS: 25
📊 Total frames: 375 (Duration: 15.0 seconds)
🚀 Starting player tracking with complete re-identification...
📊 Frame 0/375 (0.0%)
📊 Frame 10/375 (2.7%) - ETA: 10.2 min
...
✅ Processing completed successfully!
📈 Complete Processing Statistics:
   Frames processed: 375/375
   Processing time: 9.2 minutes
   Average time per frame: 1.47 seconds
   Total unique players tracked: 58
   Total re-identifications: 105
```

## 🤝 Contributing

This is a project submission. For questions or issues, please refer to the technical report in `docs/REPORT.md`.

## 📄 License

This project is created for educational and evaluation purposes.

## 📞 Support

For technical support or questions about this implementation, please refer to the documentation in the `docs/` folder.

---

**Note**: This system is designed to work with the provided YOLO model and video format. For different inputs, you may need to adjust the configuration parameters. 
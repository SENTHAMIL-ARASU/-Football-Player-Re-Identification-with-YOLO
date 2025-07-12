# player_reid.py

import cv2
import numpy as np
from ultralytics import YOLO
import math
import os

# Load YOLOv11 model
model_path = "best (2).pt"  # Updated to match your actual model file
if not os.path.exists(model_path):
    print(f"❌ Error: Model file '{model_path}' not found!")
    exit(1)

print(f"📦 Loading model from {model_path}...")
model = YOLO(model_path)

# Load input video
video_path = "15sec_input_720p.mp4"
if not os.path.exists(video_path):
    print(f"❌ Error: Input video '{video_path}' not found!")
    print("Please place your input video file in the same directory.")
    exit(1)

print(f"🎥 Loading video from {video_path}...")
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"📐 Video dimensions: {width}x{height}, FPS: {fps}")

# Prepare output video
output_path = "output.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Tracking setup
next_id = 0
trackers = {}  # {id: centroid}

def get_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

frame_id = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("🚀 Starting player tracking...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Show progress
    if frame_id % 30 == 0:  # Show progress every 30 frames
        progress = (frame_id / total_frames) * 100
        print(f"📊 Processing frame {frame_id}/{total_frames} ({progress:.1f}%)")

    # Run detection
    detections = model(frame)[0]
    current_centroids = []
    current_boxes = []

    for box in detections.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls != 0 or conf < 0.3:  # Only track players (class 0)
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        centroid = get_centroid((x1, y1, x2, y2))

        current_centroids.append(centroid)
        current_boxes.append((x1, y1, x2, y2))

    assigned_ids = []
    new_trackers = {}

    for i, centroid in enumerate(current_centroids):
        min_dist = 50
        assigned_id = None

        for tid, tcentroid in trackers.items():
            dist = euclidean(centroid, tcentroid)
            if dist < min_dist and tid not in assigned_ids:
                assigned_id = tid
                min_dist = dist

        if assigned_id is None:
            assigned_id = next_id
            next_id += 1

        new_trackers[assigned_id] = centroid
        assigned_ids.append(assigned_id)

        # Draw bounding box & ID
        x1, y1, x2, y2 = current_boxes[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {assigned_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    trackers = new_trackers
    out.write(frame)
    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ DONE! Output saved as {output_path}")
print(f"🎯 Total players tracked: {next_id}")

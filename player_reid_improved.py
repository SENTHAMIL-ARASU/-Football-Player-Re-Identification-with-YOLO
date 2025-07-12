# player_reid_improved.py - Enhanced version with better error handling

import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
import time
import sys

def main():
    # Load YOLOv11 model
    model_path = "best (2).pt"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        return False

    print(f"📦 Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

    # Load input video
    video_path = "15sec_input_720p.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Error: Input video '{video_path}' not found!")
        return False

    print(f"🎥 Loading video from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video file!")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📐 Video dimensions: {width}x{height}, FPS: {fps}")
    print(f"📊 Total frames: {total_frames} (Duration: {total_frames/fps:.1f} seconds)")

    # Prepare output video
    output_path = "output_improved.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("❌ Error: Could not create output video!")
        cap.release()
        return False

    # Tracking setup
    next_id = 0
    trackers = {}  # {id: centroid}

    def get_centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def euclidean(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    frame_id = 0
    start_time = time.time()
    last_progress_time = start_time

    print("🚀 Starting player tracking...")
    print("💡 Press Ctrl+C to stop processing (output will be saved)")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"✅ Reached end of video at frame {frame_id}")
                break

            # Show progress every 10 frames or every 5 seconds
            current_time = time.time()
            if frame_id % 10 == 0 or (current_time - last_progress_time) > 5:
                progress = (frame_id / total_frames) * 100
                elapsed = current_time - start_time
                if frame_id > 0:
                    avg_time_per_frame = elapsed / frame_id
                    remaining_frames = total_frames - frame_id
                    eta = remaining_frames * avg_time_per_frame
                    print(f"📊 Frame {frame_id}/{total_frames} ({progress:.1f}%) - ETA: {eta/60:.1f} min")
                else:
                    print(f"📊 Frame {frame_id}/{total_frames} ({progress:.1f}%)")
                last_progress_time = current_time

            # Run detection with error handling
            try:
                detections = model(frame)[0]
            except Exception as e:
                print(f"⚠️  Error in detection at frame {frame_id}: {e}")
                # Write the original frame without annotations
                out.write(frame)
                frame_id += 1
                continue

            current_centroids = []
            current_boxes = []

            # Process detections
            for box in detections.boxes:
                try:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls != 0 or conf < 0.3:  # Only track players (class 0)
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    centroid = get_centroid((x1, y1, x2, y2))

                    current_centroids.append(centroid)
                    current_boxes.append((x1, y1, x2, y2))
                except Exception as e:
                    print(f"⚠️  Error processing detection: {e}")
                    continue

            # Track players
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
                try:
                    x1, y1, x2, y2 = current_boxes[i]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID: {assigned_id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                except Exception as e:
                    print(f"⚠️  Error drawing annotation: {e}")

            trackers = new_trackers
            
            # Write frame
            try:
                out.write(frame)
            except Exception as e:
                print(f"❌ Error writing frame {frame_id}: {e}")
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Final statistics
        total_time = time.time() - start_time
        print(f"\n📈 Processing Statistics:")
        print(f"   Frames processed: {frame_id}/{total_frames}")
        print(f"   Processing time: {total_time/60:.1f} minutes")
        print(f"   Average time per frame: {total_time/frame_id:.2f} seconds")
        print(f"   Players tracked: {next_id}")
        print(f"   Output saved as: {output_path}")
        
        if frame_id == total_frames:
            print("✅ Processing completed successfully!")
        else:
            print(f"⚠️  Processing incomplete ({frame_id}/{total_frames} frames)")
        
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 
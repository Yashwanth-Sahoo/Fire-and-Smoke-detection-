import os
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load your custom trained model weights
model_path = "/FIRE AND SMOKE DETECTION/runs/detect/fire_smoke/yolov8n_baseline/weights/best.pt"
model = YOLO(model_path)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Parameters for finding corners to track inside bounding boxes
feature_params = dict(
    maxCorners=20,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Video Input Path
video_path = input("🎥 Paste the path to your video: ").strip().replace("'", "").replace('"', "")
if not os.path.exists(video_path):
    print("❌ Video path invalid!")
    exit()

# 🚀 SETUP RUNS DESTINATION DIRECTORY
# This replicates the standard YOLO output architecture
base_output_dir = "/Users/yashwanth/Downloads/FIRE AND SMOKE DETECTION/runs/detect/fast_samvid"
os.makedirs(base_output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Construct the output path inside your new runs folder
video_filename = os.path.basename(video_path)
out_path = os.path.join(base_output_dir, video_filename)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

frame_count = 0
detection_interval = 10  # Run YOLO every 10 frames

prev_gray = None
tracked_objects = []

print(f"⚡ Processing video with Hybrid YOLO + Optical Flow...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # CASE A: Time for YOLO Detection (Every 10th Frame)
    if frame_count % detection_interval == 0 or not tracked_objects:
        tracked_objects = []
        
        # 🟢 YOUR MODEL IS ACTIVATED HERE
        results = model.predict(source=frame, conf=0.25, verbose=False, device="mps")
        boxes = results[0].boxes
        
        for box in boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            cls_id = int(box.cls[0])
            label = "smoke" if cls_id == 0 else "fire"
            
            mask = np.zeros_like(frame_gray)
            mask[y1:y2, x1:x2] = 255
            
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            
            if p0 is not None:
                tracked_objects.append({
                    'box': [x1, y1, x2, y2],
                    'class': label,
                    'points': p0
                })
                
    # CASE B: Track using Optical Flow (Intermediate Frames)
    elif prev_gray is not None and len(tracked_objects) > 0:
        updated_tracked_objects = []
        
        for obj in tracked_objects:
            p0 = obj['points']
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
            
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            if len(good_new) > 0:
                shifts = good_new - good_old
                median_shift = np.median(shifts, axis=0)
                dx, dy = int(median_shift[0]), int(median_shift[1])
                
                x1, y1, x2, y2 = obj['box']
                new_box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
                
                updated_tracked_objects.append({
                    'box': new_box,
                    'class': obj['class'],
                    'points': good_new.reshape(-1, 1, 2)
                })
                
        tracked_objects = updated_tracked_objects

    # DRAW THE RESULTS
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj['box']
        label = obj['class']
        color = (0, 255, 0) if label == "smoke" else (0, 0, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)
    prev_gray = frame_gray.copy()
    frame_count += 1

cap.release()
out.release()
print(f"🎉 Done! Fast-tracked video saved inside: {out_path}")

# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# 1. Load your freshly trained custom model weights from your local directory
model_path = "/FIRE AND SMOKE DETECTION/runs/detect/fire_smoke/yolov8n_baseline/weights/best.pt"
model = YOLO(model_path)

print("🚀 Starting object detection on test images...")

# 2. Run prediction on your local test images folder
results = model.predict(
    source="/Users/yashwanth/Downloads/FIRE AND SMOKE DETECTION/dataset/data/test/images",
    conf=0.25,   # Detections must be at least 25% confident to show up
    save=True,   # Automatically saves the predicted images with bounding boxes drawn
    device="mps" # Utilizes your Mac's M5 GPU for fast processing
)

print("✅ Prediction complete!")

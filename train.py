from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="fire_smoke.yaml",
    epochs=5,
    imgsz=640,      # 🚀 Cuts pixel operations by 75%
    batch=16,       # 🚀 Fully saturates the M5's memory bandwidth
    device="mps",
    project="fire_smoke",
    name="yolov8n_baseline"
)
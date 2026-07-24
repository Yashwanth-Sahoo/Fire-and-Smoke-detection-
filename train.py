from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="fire_smoke.yaml",
    epochs=5,
    imgsz=640,
    batch=16,       
    device="mps",
    project="fire_smoke",
    name="yolov8n_baseline"
)

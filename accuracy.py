from ultralytics import YOLO

# 1. Load your trained model weights
model_path = "/Users/yashwanth/Downloads/FIRE AND SMOKE DETECTION/runs/detect/fire_smoke/yolov8n_baseline/weights/best.pt"
model = YOLO(model_path)

print("=========================================")
print("RUNNING MODEL VALIDATION ACCURACY TEST...")
print("=========================================")

# 2. Run validation
metrics = model.val(
    data="fire_smoke.yaml",
    imgsz=640,          
    device="mps"        
)

# 3. Print out the precise accuracy metrics
print("\n=========================================")
print("        FINAL ACCURACY METRICS           ")
print("=========================================")
print(f"Mean Average Precision (mAP@50):   {metrics.box.map50:.4f}")
print(f"Mean Average Precision (mAP@50-95): {metrics.box.map:.4f}")
print(f"Precision (All Classes):           {metrics.box.mp:.4f}")
print(f"Recall (All Classes):              {metrics.box.mr:.4f}")
print("=========================================\n")

# 4. Print individual class breakdowns (Fire vs Smoke)
print("Class-Specific mAP50:")
# Get list of class names in order
class_names = list(metrics.names.values())
# metrics.box.maps contains the mAP50 score for each individual class
for i, name in enumerate(class_names):
    class_map50 = metrics.box.maps[i]
    print(f" - {name}: {class_map50:.4f}")
import cv2
from ultralytics import YOLO

# 1. Load your custom trained model weights
# (Update this path to point to your fresh 640px/30-epoch 'best.pt' file once trained!)
model_path = "/FIRE AND SMOKE DETECTION/runs/detect/fire_smoke/yolov8n_baseline/weights/best.pt"
model = YOLO(model_path)

# 2. Initialize the live camera stream
# Index 0 targets your MacBook's built-in FaceTime HD camera
cap = cv2.VideoCapture(0)

# Check if the webcam opened properly
if not cap.isOpened():
    print("❌ Error: Could not open the webcam feed.")
    exit()

print("🟢 Live webcam feed initialized successfully!")
print("📺 A window will pop up shortly. Press 'q' on your keyboard to close it.")

# 3. Spin the infinite live processing frame loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to grab frame from webcam.")
        break

    # 4. Run real-time prediction using your M5 Mac GPU
    # - stream=True utilizes a memory generator pipeline to prevent RAM overflow
    # - conf=0.15 lowered slightly to pick up early-stage/small smoke signatures
    results = model.predict(
        source=frame,
        conf=0.15,
        device="mps",
        stream=True,
        verbose=False
    )
    
    # 5. Extract the frame with painted bounding boxes
    # Because stream=True returns a generator, we iterate once to grab the result
    for r in results:
        annotated_frame = r.plot()
        
    # 6. Display the live frame in a dedicated native window
    cv2.imshow("🔥 Live Fire & Smoke Detection Engine", annotated_frame)
    
    # 7. Safety break: If you press the 'q' key, kill the loop instantly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Clean up system hooks and close windows cleanly
cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam feed stopped and memory cleared successfully.")

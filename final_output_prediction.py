import os
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# 1. Load your custom trained model weights
model = YOLO("/Users/yashwanth/Downloads/FIRE AND SMOKE DETECTION/runs/detect/fire_smoke/yolov8n_baseline/weights/best.pt")

# 2. Input image path
my_image_path = input("📂 Please enter or paste the path to your image: ").strip()
my_image_path = my_image_path.replace("'", "").replace('"', "")

# 3. Check if the file exists before running the prediction
if os.path.exists(my_image_path):
    print(f"🔄 Processing image: {my_image_path}...")

    # 4. Run prediction
    results = model.predict(
        source=my_image_path,
        conf=0.25,
        save=True,
        name="single_predict",
        exist_ok=True,
    )

    # 🚀 THE CRITICAL FIX: Get the exact target output path dynamically from YOLO
    # results[0].path contains the original filename (e.g., 'smoke.webp')
    filename = os.path.basename(results[0].path)
    
    # YOLO automatically converts input extensions to standard image writes (.jpg) when saving boxes
    filename_jpg = os.path.splitext(filename)[0] + ".jpg"
    
    # Join it directly to the exact active output directory
    exact_output_path = os.path.join(results[0].save_dir, filename_jpg)

    # 5. Render only the live predicted asset
    if os.path.exists(exact_output_path):
        print(f"📸 Displaying current prediction: {filename_jpg}")
        img = Image.open(exact_output_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    else:
        print(f"❌ Error: Expected prediction file not found at {exact_output_path}")

else:
    print(f"❌ Error: The source file at '{my_image_path}' does not exist.")
import os
import glob
import math
import cv2
import matplotlib.pyplot as plt

# 🚀 Force macOS to open an interactive graphics window locally
plt.switch_backend('TkAgg')

# 1. Define the base directory where YOLO saves prediction outputs
detect_dir = "/FIRE AND SMOKE DETECTION/runs/detect"

# 2. Automatically find the most recent 'predict' folder (e.g., predict, predict2, etc.)
predict_folders = glob.glob(os.path.join(detect_dir, "predict*"))

if not predict_folders:
    print(f"❌ No prediction folders found in {detect_dir}")
    print("Please check where your predicted images were saved.")
    exit()

# Sort the folders so the latest run is at the end, then select it
predict_folders.sort(key=os.path.getmtime)
latest_predict_folder = predict_folders[-1]

print(f"📂 Found latest prediction folder: {latest_predict_folder}")

# 3. Gather all the processed images inside that folder
pred_imgs = glob.glob(os.path.join(latest_predict_folder, "*.jpg"))

if not pred_imgs:
    print(f"⚠️ No .jpg images found inside {latest_predict_folder}")
    exit()

# Just grab the first 10 images so your screen doesn't get overcrowded
pred_imgs = pred_imgs[:10] 
print(f"🖼️ Found {len(pred_imgs)} images to display.")

# 4. Configuration for the grid layout display
cols = 5
rows = math.ceil(len(pred_imgs) / cols)

# Dynamically set figure size based on rows
plt.figure(figsize=(15, rows * 3))

for idx, img_path in enumerate(pred_imgs):
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"⚠️ Warning: Could not read image at {img_path}. Skipping.")
        continue
        
    # Convert BGR (OpenCV default) to RGB (Matplotlib default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Position the image inside the grid layout matrix
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(img)
    plt.axis("off")

# Adjust spaces between pictures cleanly
plt.tight_layout()

print("🖼️ Displaying image grid window...")
plt.show()

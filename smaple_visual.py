import math
import random
import warnings
import cv2
import glob
import matplotlib.pyplot as plt
import os

# Force macOS to pop up an interactive window
plt.switch_backend('TkAgg') 
warnings.filterwarnings("ignore")

# ==========================================
# 1. DEFINE YOUR VISUALIZATION RECIPE
# ==========================================
def visualize_yolo_sample(image_path):
    label_path = image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    h, w, _ = img.shape
    if not os.path.exists(label_path):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    with open(label_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        cls, x, y, bw, bh = map(float, line.split())
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        
        # Color & Label setup (0 = smoke [green], 1 = fire [red])
        color = (0, 255, 0) if int(cls) == 0 else (255, 0, 0)
        label = "smoke" if int(cls) == 0 else "fire"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ==========================================
# 2. DEFINE YOUR GRID RECIPE
# ==========================================
def show_random_samples(train_images, num_samples=21):
    if not train_images:
        print("❌ No images found to display.")
        return

    sample_images = random.sample(train_images, min(num_samples, len(train_images)))
    cols = 3
    rows = math.ceil(len(sample_images) / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
    axes = axes.flatten()
    
    for ax, img_path in zip(axes, sample_images):
        img = visualize_yolo_sample(img_path)
        if img is not None:
            ax.imshow(img)
        ax.axis("off")
        
    for ax in axes[len(sample_images):]:
        ax.axis("off")
        
    plt.tight_layout()
    plt.show() 

# ==========================================
# 3. RUNNING THE SCRIPT
# ==========================================
train_images = glob.glob("dataset/data/train/images/*")

print("==============================")
print("THE SCRIPT IS RUNNING!")
print(f"Total Images Found: {len(train_images)}")
print("==============================")

# Execute the function to open the window
show_random_samples(train_images, num_samples=21)
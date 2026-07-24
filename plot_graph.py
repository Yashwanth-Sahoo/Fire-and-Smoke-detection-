import matplotlib.pyplot as plt
from PIL import Image

# Force macOS to pop up interactive windows locally
plt.switch_backend('TkAgg')

# Define the base directory path where your YOLO training runs are saved
run_dir = "/FIRE AND SMOKE DETECTION/runs/detect/fire_smoke/yolov8n_baseline"

# 1. Load and display the training results plot (Loss curves, mAP metrics, etc.)
results_path = f"{run_dir}/results.png"
try:
    img_results = Image.open(results_path)
    plt.figure(figsize=(12, 6))
    plt.imshow(img_results)
    plt.axis("off")
    plt.title("YOLOv8 Training Results Metrics", fontsize=14)
    plt.show()  # Opens the first window (close this window to view the next one)
except FileNotFoundError:
    print(f"❌ Could not find results file at: {results_path}")

# 2. Load and display the confusion matrix (Classification accuracy breakdown)
matrix_path = f"{run_dir}/confusion_matrix.png"
try:
    img_matrix = Image.open(matrix_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_matrix)
    plt.axis("off")
    plt.title("Confusion Matrix", fontsize=14)
    plt.show()  # Opens the second window
except FileNotFoundError:
    print(f"❌ Could not find confusion matrix file at: {matrix_path}")

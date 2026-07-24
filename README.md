# 🔥 Real-Time Fire & Smoke Detection System using YOLOv8

> An AI-powered computer vision system designed for early hazard detection and real-time localization of fire and smoke plumes using a custom-tuned YOLOv8 neural network and an interactive Streamlit web dashboard.

---

## 💡 About

Early identification of fire outbreaks and smoke accumulation is critical to preventing catastrophic property damage, environmental devastation, and loss of life. Traditional physical sensors (such as optical smoke or thermal sensors) frequently fail in outdoor environments, high-ceiling warehouses, or large industrial facilities due to smoke dispersion delays.

This project addresses these challenges by employing deep learning computer vision algorithms to detect fire and smoke in real time. Built using the lightweight **YOLOv8 Nano** architecture, the system achieves low-latency inference on edge devices while maintaining high detection precision. An interactive **Streamlit** front-end application provides real-time monitoring across static images, video files, and live webcam feeds.

---

## ✨ Features

- 🎯 **Dual-Class Detection:** Accurately localizes and classifies both `fire` and `smoke` plumes in real time.
- 📸 **Single Image Analyzer:** Side-by-side visual comparison between original images and AI bounding box annotations.
- 🎥 **Video Stream Processor:** Frame-by-frame temporal inference for pre-recorded video assets.
- 📺 **Live Webcam Feed:** Low-latency real-time video stream detection using local camera hardware.
- 🎚️ **Dynamic Confidence Slider:** Interactive control to adjust confidence thresholds (0.05 to 1.0) on-the-fly.
- ⚡ **Hardware Acceleration:** Cloud support form Google Colab GPUs as well as Apple Silicon MPS.

---

## 🛠️ Tech Stack

- **Programming Language:** Python 3.x
- **Deep Learning Framework:** PyTorch, Ultralytics YOLOv8
- **Hardware Acceleration:** Google Colab, Apple Silicon MPS
- **Computer Vision & Image Processing:** OpenCV (`cv2`), PIL (Pillow)
- **Web Application & UI:** Streamlit
- **Data Manipulation & Metrics:** NumPy, Pandas, Matplotlib, PyYAML
- **Development Environments:** Antigravity, Google Colab 

---

## 📁 Project Structure

```text
FIRE AND SMOKE DETECTION/
├── dataset/
│   └── data/
│       ├── train/           # Training split (images & labels)
│       ├── val/             # Validation split (images & labels)
│       └── test/            # Test split (images & labels)
├── runs/
│   └── detect/
│       └── fire_smoke/
│           └── yolov8n_baseline/
│               ├── weights/
│               │   ├── best.pt              # Fine-tuned PyTorch weights
│               │   └── last.pt              # Checkpoint weights
│               ├── results.csv              # Metrics per epoch
│               ├── results.png              # Loss and mAP plots
│               └── confusion_matrix.png     # Validation matrix
├── SAMPLE_IMAGE/            # Sample test images
├── SAMPLE_VIDEO/            # Sample test videos
├── accuracy.py              # Validation accuracy evaluator script
├── app.py                   # Main Streamlit web application
├── fire_smoke.yaml          # YOLOv8 dataset configuration file
├── live_detect.py           # Real-time webcam script
├── plot_graph.py            # Training metrics visualizer
├── predict.py               # Batch inference script on test images
├── predict_video.py         # Offline video inference renderer
├── train.py                 # Model training script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fire-smoke-detection.git
cd "fire-smoke-detection"
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv myenv

# Activate on macOS/Linux
source myenv/bin/activate

# Activate on Windows
myenv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for NVIDIA GPU users:** Ensure PyTorch is installed with CUDA support:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 📊 Dataset

The model is trained on a dataset from Kaggle 


### Dataset Configuration (`fire_smoke.yaml`):
```yaml
path: /path/to/dataset/data
train: train/images
val: val/images
test: test/images

nc: 2
names: ['smoke', 'fire']
```

---

## 🚀 Usage

### 1. Train the Model
To initiate fine-tuning of the YOLOv8 model:
```bash
python train.py
```

### 2. Evaluate Model Accuracy
To calculate mAP, Precision, Recall, and class breakdown:
```bash
python accuracy.py
```

### 3. Launch Web Dashboard
To start the multi-mode Streamlit interface:
```bash
streamlit run app.py
```

### 4. Direct Webcam Detection
To run lightweight standalone webcam detection:
```bash
python live_detect.py
```

---

## 📈 Results

| Metric | Score (Epoch 30) |
| :--- | :--- |
| **mAP@50** | **74.5%** |
| **Precision (All Classes)** | **74.8%** |
| **Recall (All Classes)** | **68.0%** |
| **mAP@50-95** | **43.1%** |

*   **Training Specs:** 30 Epochs, Batch Size 16, Image Size 640x640.
*   **Performance:** Low-latency detection suitable for edge devices and automated alerting pipelines.

---

## 🖼️ Screenshots



| Dashboard Interface | Fire & Smoke Detection Bounding Boxes |
| :---: | :---: |
| *(Streamlit Dashboard UI)* - <img width="1465" height="812" alt="image" src="https://github.com/user-attachments/assets/aa0da4b4-286b-451e-8a3e-c5b3ce0136e9" />
| *(Model Annotated Predictions)* - <img width="876" height="262" alt="image" src="https://github.com/user-attachments/assets/da5124dd-ce08-4972-9284-05dc073180cf" />
|

---

## 🔮 Future Improvements

- 🚀 **NVIDIA TensorRT Deployment:** Export fine-tuned weights (`.pt` to `.engine`) for maximum throughput on NVIDIA Jetson / RTX edge devices.
- 🔔 **Automated Alarm Triggering:** Integration with Twilio / MQTT APIs to dispatch instant SMS/email alerts upon hazard detection.
- 🌐 **Multi-Camera Cloud Dashboard:** Scalable WebRTC multi-stream monitoring interface.
- 🌫️ **Enhanced Small-Smoke Detection:** Data augmentation for early-stage thin smoke detection in foggy atmospheric conditions.

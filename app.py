import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

# 1. Page Configuration Setup
st.set_page_config(
    page_title="Fire & Smoke Detection Dashboard",
    page_icon="🔥",
    layout="wide"
)

# 2. Cache the model loading step so it doesn't reload on every button click
@st.cache_resource
def load_model():
    model_path = "/FIRE AND SMOKE DETECTION/runs/detect/fire_smoke/yolov8n_baseline/weights/best.pt"
    return YOLO(model_path)

model = load_model()

# 3. Sidebar Navigation Control Center
st.sidebar.title("🛠️ Control Center")
app_mode = st.sidebar.selectbox(
    "Choose Prediction Type:",
    ["Single Image", "Video Processing", "Live Webcam Feed"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Lower the confidence slider below if the model is missing small smoke plumes.")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)

# Main Dashboard Title Layout
st.title("🔥 AI Fire & Smoke Detection Engine")
st.caption("A multi-mode computer vision app driven by custom YOLOv8 weights.")
st.markdown("---")

# ==========================================
# MODE 1: SINGLE IMAGE PREDICTION
# ==========================================
if app_mode == "Single Image":
    st.header("📸 Single Image Analyzer")
    uploaded_file = st.file_uploader("Upload an image asset...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        # Create a split side-by-side screen layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_file, use_container_width=True)
            
        with col2:
            st.subheader("AI Bounding Box Prediction")
            with st.spinner("Analyzing pixels..."):
                # Save the uploaded memory buffer byte stream to a temporary file layout
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tfile:
                    tfile.write(uploaded_file.read())
                    temp_path = tfile.name
                
                # Run prediction
                results = model.predict(source=temp_path, conf=conf_threshold, device="mps", verbose=False)
                annotated_img = results[0].plot()
                
                # Streamlit expects RGB channels, YOLO outputs BGR arrays natively
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img_rgb, use_container_width=True)
                
                os.remove(temp_path)  # Cleanup disk space

# ==========================================
# MODE 2: VIDEO PROCESSING
# ==========================================
elif app_mode == "Video Processing":
    st.header("🎥 Video Processor & Renderer")
    uploaded_video = st.file_uploader("Upload a video stream asset...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        st.info("🔄 Video uploaded successfully. Initializing temporal inference processing...")
        
        # Save memory file to hard drive so OpenCV can stream-read its properties
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_video.read())
            temp_video_path = tfile.name

        cap = cv2.VideoCapture(temp_video_path)
        
        # Stream placeholders on dashboard interface dynamically
        video_placeholder = st.empty()
        
        if st.button("🚀 Start Processing Video"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process the frame on your M5 GPU layout
                results = model.predict(source=frame, conf=conf_threshold, device="mps", stream=True, verbose=False)
                for r in results:
                    annotated_frame = r.plot()
                
                # Convert frame channel arrays to RGB format layout
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Update dashboard display space actively frame-by-frame
                video_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                
            cap.release()
            st.success("🎉 Full video playback stream complete!")
            os.remove(temp_video_path)

# ==========================================
# MODE 3: LIVE WEBCAM FEED
# ==========================================
elif app_mode == "Live Webcam Feed":
    st.header("📺 Real-Time Webcam Stream")
    st.warning("Make sure your browser has given terminal applications permission to access your FaceTime HD Camera.")
    
    run_live = st.checkbox("🟢 Turn Camera ON / OFF")
    
    # Placeholder layout for active frames
    webcam_placeholder = st.empty()
    
    if run_live:
        # Connect locally to your Mac index 0 webcam framework
        cap = cv2.VideoCapture(0)
        
        while run_live:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to extract active video frame matrices from camera asset.")
                break
                
            # Compute bounding box layouts asynchronously via generator
            results = model.predict(source=frame, conf=conf_threshold, device="mps", stream=True, verbose=False)
            for r in results:
                annotated_frame = r.plot()
                
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame actively on webpage template area
            webcam_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()
        webcam_placeholder.empty()
        st.info("🛑 Webcam connection closed safely.")

import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 🚀 Force macOS to handle interactive window displays cleanly
plt.switch_backend('TkAgg')

# 1. Load your custom trained model weights
model_path = "/Users/yashwanth/Downloads/FIRE AND SMOKE DETECTION/runs/detect/fire_smoke/yolov8n_baseline/weights/best.pt"
model = YOLO(model_path)

# 2. Input path to your source video file
video_path = input("🎥 Please enter or paste the path to your video file: ").strip()
video_path = video_path.replace("'", "").replace('"', "")

# 3. Verify the file exists before running the heavy inference engine
if os.path.exists(video_path):
    print(f"🔄 Processing video: {os.path.basename(video_path)}...")
    print("🚀 Running object detection across frames (this may take a moment)...")

    # 4. Run prediction specifically configured for video stream tracking
    # - save=True automatically exports a processed .avi/.mp4 file with boxes drawn
    # - device="mps" passes calculations to your MacBook M5 GPU cores
    results = model.predict(
        source=video_path,
        conf=0.5,
        save=True,
        device="mps",
        name="video_predict",
        exist_ok=True
    )

    # 5. Extract and print the exact destination location of your output video
    # YOLO automatically lets you know where it saves the file via results[0].save_dir
    saved_dir = results[0].save_dir
    print(f"\n✅ Video processing complete!")
    print(f"📂 Check your processed video output file inside: {saved_dir}\n")

    # ========================================================
    # OPTIONAL: LIVE WINDOW PLAYER (Plays the output video back)
    # ========================================================
    play_now = input("📺 Would you like to play the processed video right now? (y/n): ").strip().lower()
    
    if play_now == 'y':
        # Find the saved video file (YOLO usually saves it with the original filename or as an .avi/.mp4)
        # We loop through the directory assets to target the video output cleanly
        for file in os.listdir(saved_dir):
            if file.endswith(('.mp4', '.avi', '.mov')):
                output_video_path = os.path.join(saved_dir, file)
                
                # Use OpenCV's video capture object to read the generated file
                cap = cv2.VideoCapture(output_video_path)
                
                print("Press the 'q' key on your keyboard at any time to close the video player.")
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break  # Video is finished
                    
                    # Open standard OpenCV live window
                    cv2.imshow("🔥 Fire & Smoke Detection Video Player", frame)
                    
                    # Play at standard frame rate interval (~30fps), check if user presses 'q' to quit
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                        
                cap.release()
                cv2.destroyAllWindows()
                break
else:
    print(f"❌ Error: The video file at '{video_path}' does not exist. Please check the path and try again!")
import cv2
import numpy as np
from ultralytics import YOLO
import os

# --- CONFIGURATION ---
VIDEO_PATH = "my_cctv_video.mp4"
# We assume the average person in the video is roughly 170cm (5ft 7in)
ASSUMED_REAL_HEIGHT_CM = 170.0 

print("------------------------------------------------")
print("AUTO-CALIBRATION TOOL")
print("------------------------------------------------")

if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: Video file '{VIDEO_PATH}' not found!")
    exit()

print("Loading AI to measure pixel height...")
try:
    human_detector = YOLO('yolov8n.pt')
except:
    print("Error: Could not load YOLO. Run: pip install ultralytics")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)
heights_collected = []

print("Scanning video... Please wait.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Run Detection
    results = human_detector(frame, classes=0, verbose=False)
    
    for result in results:
        for box in result.boxes:
            # Get the box coordinates
            _, y1, _, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Calculate height in pixels
            pixel_h = y2 - y1
            
            # Filter out small noise (must be a real person)
            if pixel_h > 50:
                heights_collected.append(pixel_h)

    # Stop after we have enough samples (100 frames is plenty)
    if len(heights_collected) > 100:
        break

cap.release()

if len(heights_collected) == 0:
    print("FAILED: No human detected in the video.")
    print("Try recording a video where the person is closer or clearer.")
else:
    # 1. Calculate Average Pixel Height
    avg_pixel_height = sum(heights_collected) / len(heights_collected)
    
    # 2. Calculate the Magic Ratio
    # Formula: Ratio = Real_CM / Pixel_Height
    final_ratio = ASSUMED_REAL_HEIGHT_CM / avg_pixel_height
    
    print("\n------------------------------------------------")
    print("CALIBRATION SUCCESSFUL!")
    print(f"Average Pixel Height Found: {avg_pixel_height:.2f} px")
    print("------------------------------------------------")
    print(f"YOUR MAGIC NUMBER IS:  {final_ratio:.4f}")
    print("------------------------------------------------")
    print(f"ACTION: Go to '3_run_system.py' and change line 15 to:")
    print(f"PIXEL_TO_CM_RATIO = {final_ratio:.4f}")
    print("------------------------------------------------")
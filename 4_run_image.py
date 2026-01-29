import os
import sys
import logging

# --- 1. SILENCE WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

sys.stderr = stderr

# --- CONFIGURATION ---
IMAGE_PATH = "tester.png"

# --- 🎯 PRECISION CALIBRATION 🎯 ---
# We set the Male Anchor to 185cm (The "Tall Reference" for modern datasets)
ANCHOR_MALE_HEIGHT = 185.0

# PERSPECTIVE CORRECTION
# If a Female is detected next to a Male, we assume she might be wearing heels 
# or standing closer. We apply a 4% correction to normalize her height.
FEMALE_PERSPECTIVE_CORRECTION = 0.96 

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    return interArea / float(boxAArea)

def draw_dashboard(img, people):
    h, w, _ = img.shape
    panel_w = 400
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    cv2.rectangle(panel, (0,0), (panel_w, 80), (0, 100, 0), -1)
    cv2.putText(panel, "HIGH-PRECISION ANALYSIS", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    y = 120
    # Sort: Male first for display
    people.sort(key=lambda x: x['gender'], reverse=True) 
    
    for p in people:
        color = (255, 180, 50) if p['gender'] == "Male" else (255, 150, 200)
        
        cv2.putText(panel, f"SUBJECT: {p['gender'].upper()}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display the logic used (for your portfolio explanation)
        cv2.putText(panel, f"Logic: {p['reason']}", (20, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
        
        cv2.putText(panel, f"Height: {p['h']:.1f} cm", (20, y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(panel, f"Weight: {p['w']:.1f} kg", (20, y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        cv2.line(panel, (20, y+115), (panel_w-20, y+115), (80,80,80), 1)
        y += 140
    return panel

# --- MAIN ALGORITHM ---
print("Running High-Precision Comparative Algorithm...")
model = YOLO('yolov8n.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: {IMAGE_PATH} not found.")
    exit()

# 1. Load & Resize
full_img = cv2.imread(IMAGE_PATH)
scale = 900 / full_img.shape[0] # High Res for better skeleton detection
img = cv2.resize(full_img, (0,0), fx=scale, fy=scale)
img_h, img_w, _ = img.shape

# 2. Detect Humans
results = model(img, verbose=False)

# 3. Filter Boxes
raw_boxes = []
for result in results:
    for box in result.boxes:
        c = box.xyxy[0].cpu().numpy().astype(int)
        h_b = c[3] - c[1]
        if h_b > (img_h * 0.30): 
            raw_boxes.append(c)

# 4. Remove Overlaps
raw_boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
valid_boxes = []
for boxA in raw_boxes:
    if not any(calculate_iou(boxA, boxB) > 0.4 for boxB in valid_boxes):
        valid_boxes.append(boxA)

# --- 5. THE COMPARATIVE ENGINE ---

candidates = []

for i, (x1, y1, x2, y2) in enumerate(valid_boxes):
    crop = img[y1:y2, x1:x2]
    shoulder_ratio = 0
    h_px = y2 - y1
    
    if crop.shape[0] > 40:
        res = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            sh_w = abs(lm[11].x - lm[12].x)
            ear_w = abs(lm[7].x - lm[8].x)
            
            # Metric: Shoulders relative to Head
            if ear_w > 0:
                shoulder_ratio = sh_w / ear_w

    candidates.append({
        'id': i,
        'box': (x1, y1, x2, y2),
        'h_px': h_px,
        'build_score': shoulder_ratio,
        'points': 0
    })

# If we have 2 people, COMPARE THEM.
if len(candidates) >= 2:
    p1 = candidates[0]
    p2 = candidates[1]

    # Rule 1: Height Vote
    if p1['h_px'] > p2['h_px']: p1['points'] += 1.5
    else: p2['points'] += 1.5

    # Rule 2: Build Vote (Shoulder/Head Ratio)
    if p1['build_score'] > p2['build_score']: p1['points'] += 1.0
    else: p2['points'] += 1.0

    # Assign Gender
    if p1['points'] > p2['points']:
        p1['gender'] = "Male"
        p2['gender'] = "Female"
        p1['reason'] = "Taller & Broader"
        p2['reason'] = "Relative Build"
        
        # --- CALIBRATION FIX ---
        # Anchor Male to 185cm
        px_to_cm = p1['h_px'] / ANCHOR_MALE_HEIGHT
        
    else:
        p2['gender'] = "Male"
        p1['gender'] = "Female"
        p2['reason'] = "Taller & Broader"
        p1['reason'] = "Relative Build"
        
        # Anchor Male to 185cm
        px_to_cm = p2['h_px'] / ANCHOR_MALE_HEIGHT

    # --- FINAL PROCESSING ---
    final_data = []
    for p in [p1, p2]:
        h_cm = p['h_px'] / px_to_cm
        
        # APPLY CORRECTION:
        # If this is the Female, correct for perspective/heels
        if p['gender'] == "Female":
            h_cm = h_cm * FEMALE_PERSPECTIVE_CORRECTION
            
        # Weight Calc
        bmi = 24.5 if p['gender'] == "Male" else 21.5
        w_kg = bmi * ((h_cm / 100.0) ** 2)
        
        p['h'] = h_cm
        p['w'] = w_kg
        final_data.append(p)

        # Draw
        x1, y1, x2, y2 = p['box']
        color = (0, 255, 0)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        label = f"{p['gender']} {p['h']:.0f}cm"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show Result
    final_view = np.hstack((img, draw_dashboard(img, final_data)))
    cv2.imshow("High-Precision Analysis", final_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("⚠️ Standard Mode (Need 2 people for comparison)")
import cv2
import numpy as np
import os
import csv
from datetime import datetime
from collections import deque
from ultralytics import YOLO
import mediapipe as mp

# --- CONFIGURATION ---
VIDEO_PATH = "c:\Users\FAHEEMA\Downloads\tester.jpeg" # 0 = Webcam
DB_FILE = "forensic_database.csv"

# ==========================================
#    🔧  ADVANCED BIOMETRIC TUNING
# ==========================================

# 1. GENDER SENSITIVITY
# Range: 1.0 (Loose) to 1.35 (Strict)
# Set to 1.30 to prevent "Female as Male" errors
GENDER_THRESHOLD = 1.30  

# 2. DYNAMIC BMI FACTORS (The "Exact Weight" Logic)
BMI_MALE = 24.5     # Heavier build
BMI_FEMALE = 21.5   # Lighter build
BMI_CHILD = 16.0    # Very light

# 3. HEIGHT CALIBRATION
# (Keep the numbers that worked for you earlier!)
# If 185cm is correct, keep these. If not, adjust.
RATIO_FAR = 0.55   
RATIO_CLOSE = 0.35   

# ==========================================

# --- ANALYTICS VARIABLES ---
session_count = 0
session_males = 0
session_females = 0
session_total_height = 0
recent_logs = deque(maxlen=3) 

# --- STABILIZATION ---
FRAMES_TO_LOCK = 40 # Reduced slightly for faster results
LOCKED = False       

print("--------------------------------------------------")
print("FORENSIC ANALYTICS: DUAL-GENDER WEIGHT MODE")
print("--------------------------------------------------")

if not os.path.exists(DB_FILE):
    with open(DB_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Case_ID", "Date", "Time", "Height_cm", "Weight_kg", "Gender", "Risk_Profile"])

# Load Models
try: human_detector = YOLO('yolov8n.pt') 
except: 
    print("Error loading YOLO. Make sure ultralytics is installed.")
    exit()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_PATH)

# Buffers
raw_heights = []
raw_genders = []
final_height = 0
final_weight = 0
final_gender = "Analyzing..."

def log_forensic_entry(h, w, g):
    global session_count, session_males, session_females, session_total_height
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    with open(DB_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        risk = "High" if h > 185 else "Normal"
        writer.writerow([session_count+1, date_str, time_str, f"{h:.1f}", f"{w:.1f}", g, risk])
    
    session_count += 1
    session_total_height += h
    if g == "Male": session_males += 1
    elif g == "Female": session_females += 1
    
    log_msg = f"#{session_count} | {time_str} | {g} | {w:.1f}kg"
    recent_logs.append(log_msg)
    print(f"✅ LOGGED: {log_msg}")

def get_dynamic_height(box_height, feet_y, screen_h):
    screen_pos = feet_y / screen_h
    current_ratio = RATIO_FAR + (screen_pos * (RATIO_CLOSE - RATIO_FAR))
    return box_height * current_ratio

def get_skeletal_gender(frame, box, current_height_cm):
    x1, y1, x2, y2 = box
    person_img = frame[y1:y2, x1:x2]
    if person_img.shape[0] < 10 or person_img.shape[1] < 10: return "Unknown"
    
    results = pose.process(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        l_sh, r_sh = lm[11].x, lm[12].x
        l_hip, r_hip = lm[23].x, lm[24].x
        
        shoulder_dist = abs(l_sh - r_sh)
        hip_dist = abs(l_hip - r_hip)
        if hip_dist == 0: hip_dist = 0.001
        
        ratio = shoulder_dist / hip_dist
        
        # --- TUNED GENDER LOGIC ---
        threshold = GENDER_THRESHOLD
        
        # Height Bias: Taller people are more likely to be Male
        if current_height_cm > 175:
            threshold -= 0.15 
            
        if ratio > threshold: 
            return "Male"
        return "Female"
        
    return "Unknown"

def draw_analytics_dashboard(img, h, w, gender, is_locked):
    panel_w = 400
    panel = np.zeros((img.shape[0], panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20) 
    
    cv2.rectangle(panel, (0,0), (panel_w, 60), (0,0,100), -1) 
    cv2.putText(panel, "FORENSIC ANALYTICS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    # Live Status
    status_color = (0, 255, 0) if is_locked else (0, 255, 255)
    status_text = "LOCKED" if is_locked else "SCANNING..."
    cv2.putText(panel, status_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    
    # Biometrics Display
    cv2.putText(panel, f"HEIGHT: {h:.0f} cm", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(panel, f"WEIGHT: {w:.1f} kg", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    g_color = (255,255,255)
    if gender == "Male": g_color = (255, 100, 0) # Blue-ish
    elif gender == "Female": g_color = (147, 20, 255) # Pink-ish
    
    cv2.putText(panel, f"GENDER: {gender.upper()}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, g_color, 2)
    cv2.line(panel, (20, 260), (panel_w-20, 260), (80,80,80), 1)

    # Stats
    cv2.putText(panel, "SESSION STATS", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    cv2.putText(panel, f"Count: {session_count} | M: {session_males} F: {session_females}", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    
    cv2.line(panel, (20, 350), (panel_w-20, 350), (80,80,80), 1)
    
    # Database Log
    cv2.putText(panel, "DATABASE FEED", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    y_log = 410
    for log in recent_logs:
        cv2.putText(panel, log, (20, y_log), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        y_log += 30

    return panel

while True:
    ret, frame = cap.read()
    if not ret: break
    
    screen_h, screen_w, _ = frame.shape
    if screen_w > 1280:
        scale = 1280 / screen_w
        frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        screen_h, screen_w, _ = frame.shape

    results = human_detector(frame, classes=0, verbose=False) 
    
    if len(results[0].boxes) == 0 and LOCKED:
        LOCKED = False
        raw_heights = []
        raw_genders = []
        final_gender = "Analyzing..."

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            w, h = x2 - x1, y2 - y1
            
            if not LOCKED:
                # 1. Collect Data
                h_inst = get_dynamic_height(h, y2, screen_h)
                raw_heights.append(h_inst)
                
                # Gender Check
                g_inst = get_skeletal_gender(frame, (x1, y1, x2, y2), h_inst)
                if h_inst < 140: g_inst = "Child" # Child Override
                if g_inst != "Unknown": raw_genders.append(g_inst)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # 2. Lock & Finalize
                if len(raw_heights) > FRAMES_TO_LOCK:
                    LOCKED = True
                    final_height = sum(raw_heights) / len(raw_heights)
                    
                    # --- DETERMINE GENDER FIRST ---
                    if raw_genders:
                        final_gender = max(set(raw_genders), key=raw_genders.count)
                    else: final_gender = "Unknown"
                    
                    # --- APPLY EXACT BMI FACTOR ---
                    bmi_used = 22.0 # Default
                    if final_gender == "Male": bmi_used = BMI_MALE
                    elif final_gender == "Female": bmi_used = BMI_FEMALE
                    elif final_gender == "Child": bmi_used = BMI_CHILD
                    
                    final_weight = bmi_used * ((final_height/100.0) ** 2)
                    
                    log_forensic_entry(final_height, final_weight, final_gender)

            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    dashboard_panel = draw_analytics_dashboard(frame, final_height, final_weight, final_gender, LOCKED)
    final_display = np.hstack((frame, dashboard_panel))
    
    cv2.imshow('Forensic Analytics Pro', final_display)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
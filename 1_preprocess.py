import os
import cv2
import numpy as np

# --- CONFIGURATION ---
DATASET_PATH = "dataset"
OUTPUT_PATH = "gei_output"
IMG_SIZE = (64, 64)
TARGET_ANGLE = "090"  # Side view

def get_images_from_path(path):
    valid_images = []
    try:
        items = sorted(os.listdir(path))
        for item in items:
            full_path = os.path.join(path, item)
            # Check inside folders
            if os.path.isdir(full_path):
                if TARGET_ANGLE in item:
                    sub_images = sorted(os.listdir(full_path))
                    for sub_img in sub_images:
                        if sub_img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            valid_images.append(os.path.join(full_path, sub_img))
            # Check files directly
            elif os.path.isfile(full_path):
                if TARGET_ANGLE in item and item.lower().endswith(('.png', '.jpg', '.jpeg')):
                    valid_images.append(full_path)
    except Exception: pass
    return valid_images

def generate_gei(sequence_path):
    image_paths = get_images_from_path(sequence_path)
    if not image_paths: return None
    
    silhouettes = []
    for img_path in image_paths:
        try:
            frame = cv2.imread(img_path, 0)
            if frame is None: continue
            coords = cv2.findNonZero(frame)
            if coords is None: continue
            x, y, w, h = cv2.boundingRect(coords)
            cropped = frame[y:y+h, x:x+w]
            resized = cv2.resize(cropped, IMG_SIZE)
            silhouettes.append(resized)
        except: continue
    
    if len(silhouettes) < 5: return None 
    gei = np.mean(silhouettes, axis=0).astype(np.uint8)
    return gei

def process_dataset():
    if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: '{DATASET_PATH}' folder not found.")
        return

    subjects = os.listdir(DATASET_PATH)
    print(f"Scanning {len(subjects)} subjects...")
    count = 0
    for subject in subjects:
        subj_path = os.path.join(DATASET_PATH, subject)
        if not os.path.isdir(subj_path): continue
        sequences = os.listdir(subj_path)
        for seq in sequences:
            seq_path = os.path.join(subj_path, seq)
            if not os.path.isdir(seq_path): continue
            gei = generate_gei(seq_path)
            if gei is not None:
                save_name = f"{subject}_{seq}.png"
                cv2.imwrite(os.path.join(OUTPUT_PATH, save_name), gei)
                count += 1
    print(f"PREPROCESSING COMPLETE. Generated {count} Shadow Images.")

if __name__ == "__main__":
    process_dataset()
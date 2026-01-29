import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATA_PATH = "gei_output"

def load_data():
    data = []
    labels = []
    if not os.path.exists(DATA_PATH):
        print("ERROR: Run 1_preprocess.py first!")
        exit()
    files = os.listdir(DATA_PATH)
    for f in files:
        if not f.endswith(".png"): continue
        try:
            subject_id = int(f.split("_")[0]) - 1 
            img = cv2.imread(os.path.join(DATA_PATH, f), 0)
            img = img / 255.0
            img = np.expand_dims(img, axis=-1)
            data.append(img)
            labels.append(subject_id)
        except: pass
    return np.array(data), np.array(labels)

X, y = load_data()
if len(X) == 0:
    print("ERROR: No images found.")
    exit()

num_classes = len(np.unique(y))
print(f"Training on {len(X)} samples for {num_classes} subjects.")

y_cat = to_categorical(y, num_classes=num_classes)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
model.save("gait_model.h5")
print("MODEL SAVED: 'gait_model.h5'")
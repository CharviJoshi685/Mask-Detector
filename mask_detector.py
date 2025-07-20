# mask_detector.py – Real-time face mask detection using webcam

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model and face detector
model = load_model("models/mask_model.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Labels and colors
LABELS = ["Mask", "No Mask"]
COLORS = [(0, 255, 0), (0, 0, 255)]

# Start webcam
cap = cv2.VideoCapture(0)
print("🎥 Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)[0]
        label_idx = np.argmax(prediction)
        label = LABELS[label_idx]
        color = COLORS[label_idx]

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("SmartMask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

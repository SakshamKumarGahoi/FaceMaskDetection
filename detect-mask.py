import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model (should be in .h5 format)
model = load_model("face_mask_detection_model.h5")

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

# Reduce resolution slightly for performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_img = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_img, (100, 100))
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 100, 100, 3))

        # Predict mask or no mask
        prediction = model.predict(reshaped_face)[0][0]

        label = "Mask" if prediction < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show frame
    cv2.imshow("Face Mask Detection", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

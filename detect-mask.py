import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

model = load_model('face_mask_detection_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  

labels = ['No Mask', 'Mask']
last_detection_time = 0
detection_interval = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_time = time.time()
    if current_time - last_detection_time > detection_interval:
        last_detection_time = current_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(face) == 0:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Face Mask Detection', frame)
        continue

    for(x, y, w, h) in face:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face = face.astype('float32') / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)[0][0]
        label = labels[int(prediction > 0.5)]
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

        cv2.putText(frame,label, (x , y - 10),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.imshow('Face Mask Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()   
cv2.destroyAllWindows()
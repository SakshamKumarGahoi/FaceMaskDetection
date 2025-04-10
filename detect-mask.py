import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('face_mask_detectortion_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  

labels = ['No Mask', 'Mask']

while True:
    ret,frame = cap
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for(x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face = face.astype('float32') / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        (mask, without_mask) = model.predict(face)[0]
        label ='Mask' if mask > without_mask else 'No Mask'
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

        cv2.putText(frame,label, (x , y - 10),
                    cv2.FONT_MONTSERRAT, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.imshow('Face Mask Detection', frame)
        
        if cv.waitkey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()   
cv2.destroyAllWindows()
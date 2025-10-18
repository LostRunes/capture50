import cv2
import os

ESP_IP = 'ESP32_IP_HERE'
PORT = 8000

# Folder to save faces
SAVE_DIR = 'faces'
PERSON_NAME = 'Abinash'  # change per person

os.makedirs(os.path.join(SAVE_DIR, PERSON_NAME), exist_ok=True)

# Use the previous TCP client code for receiving frames
# For simplicity, we’ll simulate capturing from webcam
cap = cv2.VideoCapture(0)  # replace with ESP32 TCP frames if you want

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') \
            .detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (200, 200))
        count += 1
        cv2.imwrite(f'{SAVE_DIR}/{PERSON_NAME}/{count}.jpg', face_resized)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Saved {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

    cv2.imshow('Capture Faces', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:  # collect 50 images per person
        break

cap.release()
cv2.destroyAllWindows()

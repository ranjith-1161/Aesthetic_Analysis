import cv2
from deepface import DeepFace
import numpy as np
face_cascade = cv2.CascadeClassifier('D:/python_project/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    emotion = result["dominant_emotion"]
    print(type(emotion));
    txt = emotion;
    if(txt=='sad'):
        cv2.putText(frame, "SEE THE GOOD!", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    elif(txt=='neutral'):
        cv2.putText(frame, 'GROW GRATITUDE', (355, 355), cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 25, 255), 3)
    elif (txt == 'happy'):
        cv2.putText(frame, 'KEEP CHOOSING JOY', (355, 355), cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 250, 500), 3)
    elif (txt == 'angry'):
        cv2.putText(frame, 'STAY SILENT', (355, 355), cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 250, 500), 3)
    elif (txt == 'fear'):
        cv2.putText(frame, 'FOCUS AND WIN', (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 250, 500), 3)
    elif (txt == 'surprise'):
        cv2.putText(frame, 'EVERYTHING COUNTS', (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 250, 500), 3)
    cv2.putText(frame, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
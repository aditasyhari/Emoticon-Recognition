import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('tidak dapat membuka kamera')

while True:
    ret, frame = cap.read()
    img = cv2.flip(frame, 1)

    if frame is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) != 0:

            for(x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h+10), (0, 255, 0), 2)

                face = img[y:y+h+10, x:x+w]

            result = DeepFace.analyze(face, actions = ['emotion'], enforce_detection = False)
            # print(result)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img,
                result['dominant_emotion'],
                (90, 80),
                font,
                3.0,
                (0, 0, 255),
                2,
                cv2.LINE_4
            )

    cv2.imshow('Real Time Camera', img)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
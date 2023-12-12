import cv2
from pathlib import Path

# face_detector = cv2.CascadeClassifier('./xml/haarcascade_frontalface_alt2.xml')

face_detector = cv2.CascadeClassifier('./xml/haarcascade_upperbody.xml')  # https://github.com/Shaligram1234/Python.git
print(face_detector)
for p in Path('../../images/test/').rglob('*.jpg'):
    img = cv2.imread(p.as_posix())
    img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(g, 1.5, 5)
    if len(faces):
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        cv2.imshow('aa', img)
        cv2.waitKey()

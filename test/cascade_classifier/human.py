import cv2
from pathlib import Path

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for p in Path('../../images/test/').rglob('*.jpg'):
    img = cv2.imread(p.as_posix())
    img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    boxes, weights = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
    if len(boxes):
        print(weights)
        for (x, y, w, h) in boxes:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        cv2.imshow('aa', img)
        cv2.waitKey()

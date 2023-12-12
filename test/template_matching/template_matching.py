import cv2
import numpy as np
from pathlib import Path
import argparse
import time


def main(args):
    thr = args.thr

    templates = ['./template/helmet_0.png', './template/helmet_1.png']
    test_dir = Path(args.test_dir)
    test_images = sorted(test_dir.rglob('*.jpg'), key=lambda x: x.stem)
    print(test_images)

    for path in test_images:
        start = time.time()
        img = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for t in templates:
            template = cv2.imread(t, cv2.IMREAD_GRAYSCALE)
            w, h = template.shape[::-1]
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # startX, startY = max_loc  # 만약 cv.TM_SQDIFF 혹은 cv.TM_SQDIFF_NORMED를 사용했을경우 최솟값을 사용해야한다.
            # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

            top_left = max_loc
            loc = np.where(result >= thr) # apply threshold 
            
            cnt = 0
            for pt in zip(*loc[::-1]):
                if cnt > 10:
                    continue
                cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
                cnt += 1

            bottom_right = (top_left[0] + w, top_left[1] + h)
            end = time.time()
            print()

            title = "detected" if cnt >= 10 else "No detected"
            print(f"{title} | time elapsed {end-start}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default="./images")
    parser.add_argument("--thr", type=float, default=0.83)

    args = parser.parse_args()
    main(args)
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import time
import argparse

def get_mask(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_orange = np.array([11, 150, 150])
    high_orange = np.array([13, 255, 255])
    mask = cv2.inRange(hsv_img, low_orange, high_orange)
    orange = cv2.bitwise_and(img, img, mask=mask)

    y_low, y_high, _, _ = cv2.minMaxLoc(mask.nonzero()[0])
    x_low, x_high, _, _ = cv2.minMaxLoc(mask.nonzero()[1])

    w, h = int(x_high - x_low), int(y_high - y_low)

    top_left = tuple([int(x_low), int(y_low)])
    bottom_right = tuple([int(x_high), int(y_high)])

    return (w, h, top_left, bottom_right), mask

def main(args):
    test_dir = Path(args.test_dir)
    test_images = sorted(test_dir.rglob('*.jpg'), key=lambda x: x.stem)
    
    for path in test_images:
        result = "not connected"
        start = time.time()
        img = cv2.imread(path.as_posix())
        h, w = img.shape[0], img.shape[1]
        img = img[int(0.5*h):int(0.9*h), :]

        # # denoise
        # img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21) # 너무 오래걸림

        (w, h, top_left, bottom_right), mask = get_mask(img)
        orange = cv2.bitwise_and(img, img, mask=mask)
        boxed_orange = cv2.rectangle(orange, top_left, bottom_right, (255, 255, 255), 2)
        boxed_orange = cv2.cvtColor(boxed_orange, cv2.COLOR_BGR2RGB)
        
        # crop
        if (w >= 100) and (h >= 100): 
            # aspect ratio
            if w/h >= 2.5:
                result = "connected"

            # color ratio
            orange_cropped = orange[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
            mid_color, edge_color = 0, 0
            mid_color = len(orange_cropped[:, int(0.4*w):int(0.6*w)].nonzero()[0])
            edge_color = len(orange_cropped[:, :int(0.4*w)].nonzero()[0])
            edge_color += len(orange_cropped[:, int(0.6*w):].nonzero()[0])

            # edge ratio
            cropped = img[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            edge = cv2.Canny(cropped, 350, 360)
            height, width = edge.shape
            mid_cnt, edge_cnt = 0, 0
            mid_cnt = len(edge[:, int(0.3*width):int(0.7*width)].nonzero()[0])
            edge_cnt = len(edge[:, :int(0.3*width)].nonzero()[0])
            edge_cnt += len(edge[:, int(0.7*width):].nonzero()[0])

            if mid_cnt == 0 or (edge_color/mid_color > 50):
                result = "not connected"
            elif (edge_cnt/mid_cnt) < 4 or (edge_color/mid_color < 5):
                result = "connected"
            else:
                result = "not connected"
        end = time.time()

        print("result: {} | time elapsed {}".format(result, end-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default="./images/test")
    args = parser.parse_args()
    main(args)
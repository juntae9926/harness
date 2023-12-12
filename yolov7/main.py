from pathlib import Path
import numpy as np
import cv2
import pickle
import os
import time

from collections import deque
from utils import *

class gmm_detector:
    def __init__(self):

        self.debug = True

        self.gaussian_dicts = self.get_gaussian_dicts()
        self.masks = list()
        self.coord_lists = list()

        self.k = 1
        self.target_threshold = 200
        self.train_scale = 1
        self.test_scale = 1
        self.pixel_threshold = 50 / 255  # threshold by pixel
        self.blur_threshold = 15  # threshold to remove blur
        self.small_threshold = 100
        self.correct_threshold = 2

    
    def get_gaussian_dicts(self):
        if os.path.isfile("./gaussian_dicts.pkl"):
            with open('./gaussian_dicts.pkl', 'rb') as f:
                gaussian_dicts = pickle.load(f)
        return gaussian_dicts


    def frame_test(self, cls, frame):
        scaled_img = frame_scaling(frame, self.test_scale)
        masked_img = np.zeros(shape=scaled_img.shape, dtype=np.uint8)

        for i, gaussians in enumerate(self.gaussian_dicts):
            
            img, target = test_gmm(i, scaled_img, gaussians, self.pixel_threshold, self.correct_threshold, self.k)
            if i == 0:
                blob, binary, mask, coord_array, wire = blob_detection(target, self.blur_threshold, self.small_threshold, flag=True)
            else:
                blob, binary, mask, coord_array = blob_detection(target, self.blur_threshold, self.small_threshold)
            mask_color = [[204,102,0], [0,136,255], [193,45, 100], [63,10,237]]
            img[mask == 1] = mask_color[i]
            masked_img[mask == 1] = mask_color[i]

            blob = cv2.applyColorMap(blob.astype(np.uint8), cv2.COLORMAP_JET)
            binary = cv2.applyColorMap(binary * 255, cv2.COLORMAP_JET) 
            mask = cv2.applyColorMap(mask * 255, cv2.COLORMAP_JET)
            # Morphology - dilation: 두껍게
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            dst = cv2.dilate(mask, k)
            mask = dst
            self.coord_lists.append(coord_array)

            if self.debug and (i == 3):
                show_img = frame_scaling(masked_img, test_scale=2)
                big_img = frame_scaling(img, test_scale=2)
                merged = np.hstack((big_img, show_img))
                # cv2.imshow("img", merged)
                # cv2.waitKey(1)

        mid = time.time()

        ### Apply edge detection ###
        #   Not implemented        #

        
        ### Make algorithm for belt detection ###
        #   Not implemented        #
        orange_pixel_mean = np.mean(self.coord_lists[1], axis=0)
        purple_pixel_mean = np.mean(self.coord_lists[2], axis=0)

        belt_dist = np.linalg.norm(orange_pixel_mean - purple_pixel_mean)

        if belt_dist < 100:
            belt_result = True
            # print(f"belt is connected with dist = {belt_dist}")
        else:
            belt_result = False         
            # print(f"belt is not connected with dist = {belt_dist}")

        ### Make algorithm for hook detection ###
        #   Not implemented        #
        pixel_ratio = len(self.coord_lists[3]) / (scaled_img.shape[0]*scaled_img.shape[1])
        if pixel_ratio < 0.2:
            hook_result = True
        else:
            hook_result = False
            

        # if not len(wire) == 0 and not len(self.coord_lists[3]) == 0:
        #     min_y, max_y = np.min(wire, axis=0)[0], np.max(wire, axis=0)[0]
        #     wire = np.array(wire)
        #     cross_cnt = 0
        #     for i in range(min_y, min_y+20):
        #         red_pixel_list = self.coord_lists[3][np.where(self.coord_lists[3][:,0] == i)]
        #         blue_pixel_array = np.array(wire[np.where(wire[:,0] == i)])
        #         blue_min = np.min(blue_pixel_array, axis=0)[1]
        #         blue_max = np.max(blue_pixel_array, axis=0)[1]
        #         for j in red_pixel_list:
        #             if blue_min < j[1] < blue_max:
        #                 cross_cnt += 1

        #     for i in range(max_y-20, max_y):
        #         red_pixel_list = self.coord_lists[3][np.where(self.coord_lists[3][:,0] == i)]
        #         blue_pixel_array = np.array(wire[np.where(wire[:,0] == i)])
        #         blue_min = np.min(blue_pixel_array, axis=0)[1]
        #         blue_max = np.max(blue_pixel_array, axis=0)[1]
        #         for j in red_pixel_list:
        #             if blue_min < j[1] < blue_max:
        #                 cross_cnt += 1
            
        #     if cross_cnt > 10:
        #         hook_result = True
        #         # print(f"hook is connected with {cross_cnt} pixels")
        #     else:
        #         hook_result = False
        #         print(f"hook is not connected with {cross_cnt} pixels")
        # else:
        #     hook_result = False
            # print("hook is not connected")
        self.coord_lists = []
        end = time.time()
        # print("time elapsed: {:.04f} sec | gmm: {:0.4f} sec | algorithm: {:0.4f} sec".format(end-start, mid-start, end-mid))
        
        return merged, belt_result, hook_result

if __name__ == "__main__":

    gmm_detector = gmm_detector()
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture("rtsp://admin:1234@163.239.25.37:554/video1s1_audio1")

    fps_queue = deque(maxlen=100)

    while True:
        ret, img = cap.read()
        h, w, ch = img.shape
        if ret:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t, belt_result, hook_result = gmm_detector.run(img)

            fps_queue.append(t)
            fps = 1/np.mean(fps_queue)
            if len(fps_queue) == 100:
                fps_queue.pop()


            print(f"Belt result is {belt_result}, Hook result is {hook_result} | FPS: {fps:.03} ")


from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QHBoxLayout, QPushButton, QStyle
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap

from collections import deque
import numpy as np
import threading
import sys
import cv2

from main import detector
import argparse


class Camera:
    def __init__(self):
        self.run()
    
    def run(self):
        # self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap = cv2.VideoCapture("rtsp://admin:1234@163.239.25.37:554/video1s1_audio1") 

class Worker(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, camera):
        super().__init__()
        
        self.run_flag = True
        self.frame_flag = False
    
        self.camera = camera
    
    @pyqtSlot()
    def run(self):
        while self.run_flag:
            ret, img = self.camera.cap.read()
            h, w, ch = img.shape
            if ret:
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimage = QtGui.QImage(frame, w, h, ch * w, QtGui.QImage.Format_RGB888)
                pixmap = QPixmap(qimage)
                self.current_pixmap = pixmap
                self.change_pixmap_signal.emit(img)
                
        self.camera.cap.release()

    def stop(self):
        self.run_flag = False

class App(QWidget):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Video Start")

        self.initUI()

        self.finished = True
        self.frame_window = args.fr

        self.video_queue = []
        self.fps_queue = deque(maxlen=100)
        self.frame_queue = deque(maxlen=1)
        self.result_queue = deque(maxlen=1)

        self.camera = Camera()
        self.detector = detector()
        self.pixmap = QPixmap()
        self.setGeometry(200, 200, 1300, 600)
        

    def initUI(self):

        self.height = Camera().cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = Camera().cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        self.setGeometry(200, 200, self.width*2.2, self.height*2.2)

        self.image_label = QLabel(self)
        self.frame_label = QLabel(self)

        self.belt_label = QLabel(self)
        self.belt_label.resize(self.width/2, self.height*0.1)
        self.belt_label.move(self.width+16, 10)
        self.hook_label = QLabel(self)
        self.hook_label.resize(self.width/2, self.height*0.1)
        self.hook_label.move(self.width*1.5 +16, 10)

        self.belt_label_name = QLabel("Belt", self)
        self.belt_label_name.setFont(QtGui.QFont("Times", 20))
        self.belt_label_name.move(self.width*1.25, 20)
        self.hook_label_name = QLabel("Hook", self)
        self.hook_label_name.setFont(QtGui.QFont("Times", 20))
        self.hook_label_name.move(self.width*1.7, 20)

        self.fps_label = QLabel("FPS",self)
        self.fps_label.setFont(QtGui.QFont("Times", 15))
        self.fps_label.move(1150, 550)
        
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.image_label)
        self.hbox.addWidget(self.frame_label)
        self.setLayout(self.hbox)

        self.start_btn = QPushButton(self)
        self.start_btn.resize(self.width, self.height*0.1)
        self.start_btn.move(11, 10)
        self.start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_btn.clicked.connect(self.start)

        self.green = QPixmap("./images/color/green.jpg")
        self.red = QPixmap("./images/color/red.png")

    # Start three threads
    def start(self):
        # Video streaming
        self.video_thread = Worker(self.camera)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

        # Get test-frame
        self.frame_thread = Worker(self.camera)
        self.frame_thread.change_pixmap_signal.connect(self.get_frame)
        self.frame_thread.start()

        # Run algorithm
        self.detect_thread = threading.Thread(target=self.test)
        self.detect_thread.start()

    def closeEvent(self, event):
        self.video_thread.stop()
        self.frame_thread.stop()
        self.detect_thread._stop()
        event.accept()
    
    # Get every frame(Make streaming video)
    @pyqtSlot(np.ndarray)
    def update_image(self, img):
        self.video_queue.append(img)
        img = self.make_qimage(img)
        self.image_label.setPixmap(img)
      
    # Start detect algorithm 
    def test(self):
        while True:
            if (len(self.frame_queue) > 0):
                frame = self.frame_queue.pop()
                t, belt, hook = self.detector.run(frame)
                self.fps_queue.append(t)
                self.fps = 1/np.mean(self.fps_queue)
                self.fps_label.setText("FPS: " + str(round(self.fps, 3)))
                self.draw_signal(belt, hook)
                self.finished = True

                if len(self.fps_queue) == 100:
                    self.fps_queue.pop()
                

    # Get test frame
    @pyqtSlot(np.ndarray)
    def get_frame(self, img):
        if self.finished:
            self.finished = False
            self.frame_queue.append(img)
            height, width, _ = img.shape
            img = cv2.rectangle(img, (int(width*0.2), int(height*0.3)), (int(width*0.8), int(height*0.7)), (0, 0, 0), 3)
            img = self.make_qimage(img)
            self.frame_label.setPixmap(img)


    # Make numpy image to qimage
    def make_qimage(self, img:np.ndarray):
        h, w, ch = img.shape
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimage = QtGui.QImage(frame, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pixmap = QPixmap(qimage)
        return pixmap

    # Visualize final result on screen
    def draw_signal(self, belt, hook):
        if belt and hook:
            self.belt_label.setPixmap(self.green)
            self.hook_label.setPixmap(self.green)
        
        elif belt and not hook:
            self.belt_label.setPixmap(self.green)
            self.hook_label.setPixmap(self.red)

        elif not belt and hook:
            self.belt_label.setPixmap(self.red)
            self.hook_label.setPixmap(self.green)
        
        else:
            self.belt_label.setPixmap(self.red)
            self.hook_label.setPixmap(self.red)        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fr", type=int, default=110, help="Set window size")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    a = App(args)
    a.show()
    sys.exit(app.exec_())

#!/usr/bin/env python
# coding: utf-8
# %%


import sys
import urllib.request
import threading
import cv2
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from dynamic_resize import dynamic_resize


# %%
#Hyper Parameter
video_size = (640, 480) #화면, 신호창 크기
signal_size = (190, 31) #if auto =False
ui_path = './safety_harness_belt_gui.ui'
interval = 10 #실시간 동영상과 검사프레임 사이 간격
# cap = cv2.VideoCapture(0) # if auto == True

#동적 반응 맞추기: 웹캠 이미지 크기에 맞게/ 반환 값: 신호 창 (너비, 높이) 튜플
video_size, signal_size = dynamic_resize(auto=False, video_size=video_size, ui_path=ui_path, interval=interval)

# %%
#UI파일 연결
form_class = uic.loadUiType(ui_path)[0]


# %%
#화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        
        ######## 시간 설정 ########
        self.exec_time = 1.5 #(초)
        
        ########### 크기 설정 ##########
        self.width = video_size[0]
        self.height = video_size[1]
        self.signal_width = signal_size[0]
        self.signal_height = signal_size[1]
        
        # 이미지 로드 및 크기 조정
        self.qPm = QPixmap()
        self.qPm.load('./image/camera_off.jpg') # 초기화 화면
        self.initqPm = self.qPm.scaled(self.width, self.height)
        self.qPm.load('./image/green.jpg') # 신호 녹색
        self.greenqPm = self.qPm.scaled(self.signal_width,self.signal_height)
        self.qPm.load('./image/red.png') # 신호 적색
        self.redqPm = self.qPm.scaled(self.signal_width,self.signal_height)
        
        ########### 기타 #############
        #웹 캡 실행 변수
        self.running = False
        #화면 초기화
        self.initView()
        #화면 바로 시작
        self.start()
        #창 제목 설정
        self.setWindowTitle('안전 검사')
        #창 크기 설정
#         self.setGeometry()
        
        ########시그널(위젯-기능)########
        # 버튼 기능 설정
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
         
            
    ########Method(기능)########
    
    #WebCam Run 함수
    def run(self):
        
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        #프레임 수 카운트
        fr_cnt = 0
        while self.running:
            
            fr_cnt += 1
            
            # read(): 1.grab() -> ret/ 2.retrieve() -> image
            ret, img = cap.read()
            
            if ret:

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                
                h,w,c = img.shape
                
                if fr_cnt % (fps * self.exec_time) == 0: #알고리즘이 실행 후, 검사 프레임 확인
                    
                    #검사 프레임 창에 띄우기
                    qImg_test = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)

                    pixmap_test = QtGui.QPixmap.fromImage(qImg_test)

                    self.test_lbl.setPixmap(pixmap_test)
                    
                    #벨트 및 고리 체결 확인 알고리즘
                    ############# Algorithm #############
                    """
                    input : ndarray image(검사 프레임)
                    output : return belt(boolean), hook(boolean)
                    """
                    ####################################
                    
                    #test
                    belt = False
                    hook = False
                    
                    #신호 색깔 보여주기
                    self.showResult(belt=belt, hook=hook)
                    
                #실시간 동영상 창에 띄우기
                qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)

                pixmap = QtGui.QPixmap.fromImage(qImg)

                self.view_lbl.setPixmap(pixmap)
                
            else:# 프레임이 존재하지 않을 때
                win = QtWidgets.QWidget()
                # 에러 팝업 창
                QtWidgets.QMessageBox.about(win, "Error", "이미지를 읽을 수 없습니다.")

                print("cannot read frame.")

                break
        
        #Stop -> 초기 화면
        self.initView()
        
        #free resource
        cap.release()
        print("Thread end.")
    
    #웹캠 멈춤
    def stop(self):
        self.running = False
        print("stoped..")
    
    #웹캠 시작
    def start(self):
        self.running = True
        #GUI 실행과 별도의 Thread에서 이미지를 받아 view_lbl에 띄워준다.
        th1 = threading.Thread(target=self.run)
        th1.start()
        print("started..")
    
    #프로그램 종료
    def onExit(self):
        print("exit")
        #웹캠 멈춤
        self.stop()
    
    #화면 초기화
    def initView(self):
        self.view_lbl.setPixmap(self.initqPm)
        self.test_lbl.setPixmap(self.initqPm)
        
    #녹색 표시
    def showGreen(self, label):
        label.setPixmap(self.greenqPm)
        
    #적색 표시
    def showRed(self, label):
        label.setPixmap(self.redqPm)

    #결과 표시
    def showResult(self, belt, hook):
        if belt and hook:
            #signal
            self.showGreen(self.belt_lbl)
            self.showGreen(self.hook_lbl)
            #text
            self.result_lbl.setText("통과") #텍스트 변환
            self.result_lbl.setFont(QtGui.QFont("고딕",60, QtGui.QFont.Bold)) #폰트,크기 조절
            self.result_lbl.setStyleSheet("Color : green") #글자색 변환
            self.result_lbl.setAlignment(QtCore.Qt.AlignCenter) #가운데 정렬
        elif (belt == True) and (hook == False):
            #signal
            self.showGreen(self.belt_lbl)
            self.showRed(self.hook_lbl)
            #text
            self.result_lbl.setText("벨트 O | 고리 X") #텍스트 변환
            self.result_lbl.setFont(QtGui.QFont("고딕",60, QtGui.QFont.Bold)) #폰트,크기 조절
            self.result_lbl.setStyleSheet("Color : red") #글자색 변환
        elif (belt == False) and (hook == True):
            #signal
            self.showRed(self.belt_lbl)
            self.showGreen(self.hook_lbl)
            #text
            self.result_lbl.setText("벨트 X | 고리 O") #텍스트 변환
            self.result_lbl.setFont(QtGui.QFont("고딕",60, QtGui.QFont.Bold)) #폰트,크기 조절
            self.result_lbl.setStyleSheet("Color : red") #글자색 변환
        else:
            #signal
            self.showRed(self.belt_lbl)
            self.showRed(self.hook_lbl)
            #text
            self.result_lbl.setText("벨트 X | 고리 X") #텍스트 변환
            self.result_lbl.setFont(QtGui.QFont("고딕",60, QtGui.QFont.Bold)) #폰트,크기 조절
            self.result_lbl.setStyleSheet("Color : red") #글자색 변환
                
if __name__ == "__main__" :
    
    #app 생성, 프로그램을 만들기 위한 큰 바구니 생성
        #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv) 
    
    #WindowClass의 인스턴스 생성
    myWindow = WindowClass() 

    #프로그램 화면을 보여주는 코드
    myWindow.show()
    
    #프로그램이 종료시, 웹캠도 종료
    app.aboutToQuit.connect(myWindow.onExit)
    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
        #app.exec_(): 생성만하고 종료되지 않게, 대기상태(무한루프 상태)를 만든다.
        #sys.exit(): app.exec() 종료시 -> 0, sys.exit(0): 정상 종료 
    sys.exit(app.exec_())

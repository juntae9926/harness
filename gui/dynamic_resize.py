############## 웹캠 이미지 사이즈에 따라, UI 동적 반응형 구조 ##############
import cv2
import xml.etree.ElementTree as ET  #python 내장 xml library


def dynamic_resize(ui_path, auto=False, video_size=None, vc=False, interval=10):
    
    """
    ui_path: pyqt5 UI xml 파일 주소
    auto: boolean, 웹캠 이미지 맞출지, 수동 크기 입력 받아 맞출지
    video_size: 튜플, 웹캠 이미지 (너비, 높이)
    vc: cv2.VideoCapter() 객체
    interval: 실시간 동영상과 검사프레임의 간격
    """
    
    #ui read text 모드로 xml 로드 및 객체 할당
    ui_xml = open(ui_path, 'rt', encoding='UTF8')
    ui_tree = ET.parse(ui_xml)
    
    if auto:
        # 웹캠 이미지의 가로, 세로 길이
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        width = video_size[0]
        height = video_size[1]
    
    
    ############## 동적 사이즈 맞추기 #################
    
    #실시간 동영상 너비, 높이: 웹캠 사이즈에 맞추기
    view_lbl_rect = './widget/widget[@name="view_lbl"]/property/rect/'
    x = int(ui_tree.find(f'{view_lbl_rect}x').text)
    ui_tree.find(f'{view_lbl_rect}width').text = str(width)
    ui_tree.find(f'{view_lbl_rect}height').text = str(height)

    #검사 프레임 x(가로 시작점), 너비, 높이
    test_lbl_rect = './widget/widget[@name="test_lbl"]/property/rect/'
    ui_tree.find(f'{test_lbl_rect}x').text = str(x + width + interval)
    ui_tree.find(f'{test_lbl_rect}width').text = str(width)
    ui_tree.find(f'{test_lbl_rect}height').text = str(height)

    #벨트 텍스트 x
    belt_lbl_rect = './widget/widget[@name="belt_text"]/property/rect/'
    ui_tree.find(f'{belt_lbl_rect}x').text = str(x + width + interval + 10)

    #벨트 신호 x, 너비
    ui_tree.find(f'{belt_lbl_rect}x').text = str(x + width + interval + 110)
    signal_lbl_width = int(((width - 260) / 2) - ((width - 260) / 2) % 10)
    signal_lbl_height = int(ui_tree.find(f'{belt_lbl_rect}height').text)
    ui_tree.find(f'{belt_lbl_rect}width').text = str(signal_lbl_width)

    #고리 텍스트 x
    hook_lbl_rect = './widget/widget[@name="hook_text"]/property/rect/'
    hook_text_x = x + width + interval + 110 + signal_lbl_width + 40
    ui_tree.find(f'{hook_lbl_rect}x').text = str(hook_text_x)

    #고리 신호 x, 너비
    ui_tree.find(f'{hook_lbl_rect}x').text = str(hook_text_x + 100)
    ui_tree.find(f'{hook_lbl_rect}width').text = str(signal_lbl_width)
    
    # ui_xml 수정하기
    ui_tree.write(ui_path)
    
    #웹캠 (너비, 높이), 신호 (너비, 높이 값) 튜플 반환
    return (width, height), (signal_lbl_width, signal_lbl_height)

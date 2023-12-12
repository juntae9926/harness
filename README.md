# safety-harness-detection

Safety Harness Detection is made to check whether safety-belt and safety-hook are connected.


Main algorithm can be used with a single image.


## Algorithm
<hr/>

<img src=./reference/GMM.png width="800px" height="450px">

- Gaussian Mixture Model
- Blob Detection

## Required Files
<hr/>

- gaussian_dicts.pkl
- main.py
- utils.py
- window.py

## Environment Setup
<hr/>

All implementation is created in python3.9 version Wondows
```
pip install -r requirements.txt
```
## 


## Train and Test Data Setup 
<hr/>

- `images/train`: `/mldisk/nfs_shared/exchange/안전띠검출/data/dataset/train`
- `images/test`: `/mldisk/nfs_shared/exchange/안전띠검출/data/dataset/test`

-------
## Train and inference with YoloV7

### Train
```
train: python train.py --workers 8 --device 0 --batch-size 32 --data data/harness.yaml --img 640 640 --cfg cfg/training/yolov7-harness.yaml --weights yolov7.pt --name harness-6classes --hyp data/hyp.scratch.custom.yaml
```

### inference
```
python detect.py --weights runs/train/6classes/weights/best.pt --conf 0.3 --img-size 640 --source 0
```


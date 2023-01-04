# Slim YOLOv7



## Installation

- Python 3.9

````shell
pip install -r requirements.txt
````


## Usage

````shell
python train.py --device 'cpu' --batch-size 16 --data data/CNN4VIAB.yaml --img 640 640 --cfg cfg/training/yolov7-CNN4VIAB.yaml --weights 'yolov7_training.pt' --name yolov7-CNN4VIAB --hyp data/hyp.scratch.custom.yaml --epochs 300 --adam
````


## Training

* Download MS COCO dataset images ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip). If you have previously used a different version of YOLO, we strongly recommend that you delete `train2017.cache` and `val2017.cache` files, and redownload [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip) 



## Transfer learning

[`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt) [`yolov7x_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt) [`yolov7-w6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6_training.pt) [`yolov7-e6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6_training.pt) [`yolov7-d6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6_training.pt) [`yolov7-e6e_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt)





## Inference

On video:
``` shell
python detect.py --weights best.pt --conf 0.25 --img-size 640 --source <path to video>
```

On image:
``` shell
python detect.py --weights best.pt --conf 0.25 --img-size 640 --source <path to image>
```

## Acknowledgements

- Original code: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- Original paper: [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696), [![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)


## Author
 - max-scw

## Status
active


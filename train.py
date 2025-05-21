import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml')
    # model = YOLO('ultralytics/cfg/models/v8/yolov8-ASF-P2.yaml')
    model = YOLO('yolov8-ASF-P2.yaml')
    # model.load('yolov8m.pt') # loading pretrain weights
    model.train(data='VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # resume='autodl-tmp/ultralytics-main/runs/train/yolov8m-ASF-P2-coco9/weights/last.pt', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='yolov8m-ASF-P2',
                )


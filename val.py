import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('/kaggle/input/rtdetr-weight/best (4).pt')
    # /kaggle/input/yolo-weights/weights/spdnsoap.pt
    model.val(data='/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml',
              split='test',
              imgsz=640,
              batch=16,
              # rect=False,
              #save_json=False, # if you need to cal coco metrice
              project='runs/val',
              name='yolov8n-spdnosoap',
              )
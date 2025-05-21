import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/kaggle/input/yolo-weights/weights/spdld.pt') # select your model.pt path
    model.predict(source="/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/images/test",
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  verbose = True
                  #save=True,
                  # conf=0.2,
                  #visualize=False # visualize model features maps
                )
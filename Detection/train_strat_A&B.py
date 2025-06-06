import os
from ultralytics import YOLO

def train1():
    # Load pretrained model 
    model = YOLO("yolo11n.pt")  

    # config 
    data_config = "C:/...../config_WL.yaml"

    # Strategy A
    # results = model.train(data=data_config, epochs=200, patience=10)
    # Strategy B
    # results = model.train(data=data_config, epochs=200, patience=10, erasing = ...)
    # results = model.train(data=data_config, epochs=200, patience= 10, hsv_h= ...)
    # results = model.train(data=data_config, epochs=200, patience= 10, hsv_v= ...)


def train2():
    # Load best trained model from train1()
    model = YOLO("C:/...../runs/detect/train/weights/best.pt")  # build from YAML and transfer weights

    # config 
    data_config = "C:/...../config_WARP.yaml"

    # Strategy A
    # results = model.train(data=data_config, epochs=2, patience=10)
    # Strategy B
    # results = model.train(data=data_config, epochs=200, patience=10, erasing = ...)
    # results = model.train(data=data_config, epochs=200, patience= 10, hsv_h= ...)
    # results = model.train(data=data_config, epochs=200, patience= 10, hsv_v= ...)

if __name__ == '__main__':
    train1()
    print("Training 1 done")
    train2()
    print("Finished both trainings")
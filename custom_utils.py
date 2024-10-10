import sys
import os
import torch
import cv2
import pytesseract
import numpy as np
from models.yolo import Model

yolov9_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov9'))
sys.path.insert(0, yolov9_path)

checkpoint = torch.load('yolov9_model/yolov9-c.pt')

def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(image, int(dh), int(dh), int(dw), int(dw), cv2.BORDER_CONSTANT, value=color)

def load_yolov9_model():
    model = Model()  # Initialize YOLO model
    checkpoint = torch.load('path_to_your_checkpoint.pth')
    model.load_state_dict(checkpoint['model'].state_dict(), strict=False)
    return model

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold_image

def detect_text_yolov9(image, model):
    img = letterbox(image)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0) if img.ndimension() == 3 else img
    with torch.no_grad():
        return model(img)[0]

def extract_text_from_image(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(rgb_image)

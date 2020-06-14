import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy as np
import mmcv, cv2
from utils.utils import corp_img, predict_draw
import time

# model loading
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_ft = models.mobilenet_v2(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.classifier[1] = nn.Linear(1280, 2)
print('loading model')
model_ft.load_state_dict(torch.load('./model/m'))
model_ft = model_ft.to(device)

# running camera
cap = cv2.VideoCapture(0)
frame_rate = 3
prev = 0
while True:
    time_elapsed = time.time() - prev
    ret, frame = cap.read() #Capture each frame
    if time_elapsed > 1./frame_rate:
        prev = time.time()
        img = predict_draw(model_ft, frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("Face Mask Detetor", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindow("Face Mask Detetor")

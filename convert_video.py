import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy as np
import mmcv, cv2
from utils.utils import corp_img, predict_draw
import argparse
import time

# args handle
parser = argparse.ArgumentParser()
parser.add_argument("input_video_path")
parser.add_argument("output_name")
args = parser.parse_args()

# model loading
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_ft = models.mobilenet_v2(pretrained=True)
model_ft.classifier[1] = nn.Linear(1280, 2)
print('loading model')
model_ft.load_state_dict(torch.load('./model/m'))
model_ft = model_ft.to(device)

# running camera
cap = cv2.VideoCapture(args.input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('%s.avi'%(args.output_name),fourcc, 20.0, (1280,720))
frame_rate = 20
prev = 0
while(cap.isOpened()):
    time_elapsed = time.time() - prev
    ret, frame = cap.read() #Capture each frame
    if time_elapsed > 1./frame_rate:
        prev = time.time()
        try:
            img = predict_draw(model_ft, frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            img = frame
        out.write(img)
        try:
            cv2.imshow("Face Mask Detetor", img)
        except:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows("Face Mask Detetor")

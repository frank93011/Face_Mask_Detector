import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import copy
import torch.nn.functional as F
from facenet_pytorch import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

def corp_img(img, c):
    return img[c[1]:c[3], c[0]:c[2]]

def predict_draw(model, img):
    model.eval()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, a = mtcnn.detect(img)
    if(type(boxes) is not np.ndarray):
        return img

    for i in range(len(boxes)):
        bnd = boxes[i].astype(int)
        if(bnd[3]-bnd[1] < 40):
            continue
        img2 = corp_img(img,bnd)/255
        if(len(img.shape)!=3):
            return img
        img2 = cv2.resize(img2, (128,128))
        img2 = torch.tensor(img2, dtype=torch.float)
        img2 = img2.permute(2,0,1).unsqueeze(0).to(device)
    
        cv2.rectangle(img, tuple(bnd[:2]), tuple(bnd[2:]), (0,0,255), 4)
        with torch.no_grad():
            prob = F.softmax(model(img2))
            output = prob.max(1)[1]
            # print(prob)
            if(output):
                cv2.putText(img, 'No Mask:%1f'%(float(prob[0][1])), ((bnd[0]+20), bnd[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 6)
            else:
                cv2.putText(img, 'Mask:%1f'%(float(prob[0][0])), ((bnd[0]+20), bnd[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 4)
    return img
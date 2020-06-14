# Face_Mask_Detector
Implementation of face mask detector which can quickly be deployed on local camera based on MTCNN and resnet18.

## Requirement

```bash
pip install -r requirement.txt
```

## Runs

running main.py to open front camera and begin detection
```bash
python3 main.py
```
running convert_video.py to convert normal video with detection boxes.
```bash
python3 convert_video.py --input/video/path --output_file_name
```

## Demo
![](https://i.imgur.com/4XvMYho.png)
![](https://i.imgur.com/84UOp4z.png)
![](https://i.imgur.com/O7VhqUA.png)
![](https://i.imgur.com/zxaGbzP.png)
![](https://i.imgur.com/SoVvXta.png)
![](https://i.imgur.com/9pxcX3P.png)
![](https://i.imgur.com/xUNdDGC.png)
![](demo.gif)

## Training and Dataset

### Fine-tuning of Resnet18

Google Colab notebook here: https://colab.research.google.com/drive/1tBFcS0PBl2YlGCLey7cM-oUDyfcU81y-?usp=sharing

### Dataset Credits
Training datasets came from here:
https://drive.google.com/file/d/1QspxOJMDf_rAWVV7AU_Nc0rjo1_EPEDW/view

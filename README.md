# Handwritten Word Detector wit YOLOV5
Detect handwritten words with YOLOV5

## Installation
- `git clone https://github.com/kos94ok/WordDetectorYolov5.git`
- `cd WordDetectorYolov5`
- `pip install -r requirements.txt`
-  Download [pretrained model](https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1XhAMH2OelzYhnDtfUWFi5VIQOdJzpNEs), and place the unzipped files into the ckpt directory


## Usage 
```python
import torch
import cv2
import numpy as np
import words

image = cv2.imread("test/img_24.png")

model = torch.hub.load('yolov5', 'custom', path='ckpt/best_4_finetunning.pt', source='local', device='cpu')
model.conf = 0.5
model.size = 320

predictions = model(image)
crops = predictions.crop(save=False)

#Normalization
boxes = []
for crop in crops:
  boxesList = crop['box']
  boxArray = []
  for box in boxesList:
    boxArray.append(int(box))
  boxes.append(boxArray)
  
  # Sorting words from left to right
 lines = words.sort_words(np.array(boxes))

#Show results
for line in lines:
  textImageROI = image.copy()
  for (x1, y1, x2, y2) in line:
    cv2.rectangle(image, (x1, y1), (x2, y2), 125, 2)
    cv2.imshow('Image', image)
    cv2.waitKey()
```

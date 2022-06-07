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
  
  # Sort words left-right
lines = words.sort_words(np.array(boxes))

#Show results
for line in lines:
  textImageROI = image.copy()
  for (x1, y1, x2, y2) in line:
    cv2.rectangle(image, (x1, y1), (x2, y2), 125, 2)
    cv2.imshow('Image', image)
    cv2.waitKey()

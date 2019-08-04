import cv2
import os
import numpy as np

l = []
for root, dirs, files in os.walk('feature'):
    for file in files:
        if os.path.splitext(file)[-1] == '.jpg':
            l.append(os.path.join(root, file))

for item in l:
    
    img = cv2.imread(item)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('cv', img)
    cv2.waitKey(100)
import os
import cv2
import numpy as np
path = R""
filenames = os.listdir(path)
for filename in filenames:
    filename = os.path.join(path, filename)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(filename, img)


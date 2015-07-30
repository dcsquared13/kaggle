__author__ = 'dan'

import numpy as np
import cv2

img = cv2.imread('/Users/dan/Google Drive/kaggle/retinopathy/train/99_left.jpeg')
img_r = cv2.resize(img, (150, 150))

print img_r

img_r2 = img_r.reshape(3,150,150)

print img_r2

na = np.zeros((1, 150, 150, 3))
print img_r.shape

na[0] = img_r

print na.shape

print na



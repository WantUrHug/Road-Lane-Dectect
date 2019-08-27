import cv2
import numpy as np
import os
from utils import VideoGo

test_img = "D:\\0824\\Project1\\Project1\\test_videos\\pic_001.jpg"
#im = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
im = cv2.imread(test_img)
im = im[450:, :, :]

h, w, _ = im.shape
print(h, w)
cv2.imshow("windows", im)
cv2.waitKey(500)
cv2.destroyAllWindows()
pts = np.float32([[0,0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
pts1 = np.float32([[0,0], [w/2 - 1 - 100, h - 1], [w/2 - 1 + 100, h - 1], [w - 1, 0]])

M = cv2.getPerspectiveTransform(pts, pts1)
 
dst = cv2.warpPerspective(im,M, dsize = (1280, 270))

cv2.imshow("windows", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
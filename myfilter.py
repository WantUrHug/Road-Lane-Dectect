import cv2
import os
import numpy as np
from utils import VideoGo

def fun1(input):
	'''
	:frame --> np.ndarray
	'''
	input = input[360 :, :, :]

	frame = cv2.blur(input, (4,4))
	frame = cv2.GaussianBlur(frame, (5,5), 1)
	frame = cv2.Canny(frame, 50, 150, apertureSize = 3)

	#lines = cv2.HoughLinesP(frame)
	return frame

if __name__ == "__main__":

	test_vd = "D:\\0824\\Project1\\Project1\\test_videos\\06.mp4"
	VideoGo(test_vd, fun1, 4)

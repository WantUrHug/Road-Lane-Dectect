import cv2
import os
import numpy as np

def build_test_plane(dst = None):

	a = np.zeros((640, 480), dtype = "uint8")

	width = 40
	dis = 80

	for i in range(640):
		for j in range(480):
			if (j >= dis and j < dis + width) or (j >= 480 - (dis + width) and j < 480 - dis):
				a[i, j] = 255
	cv2.imshow("WINDOW", a)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite("abstract.jpg", a)



def VideoGo(vd_path, filter_fun = None, faster = 1):
	#逐帧播放视频.
	#通过函数filter_fun，逐帧地对图片进行修改，一来可以查看不同
	#滤波条件或者变换的效果如何，二来可以观察稳定性
	#
	capture = cv2.VideoCapture(vd_path)

	nbframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

	fps = capture.get(cv2.CAP_PROP_FPS)
	#print("fps is ", fps)

	#读取下一帧
	success, frame = capture.read()

	if not filter_fun:
		while success:
			cv2.imshow("windows", frame)
			cv2.waitKey(int(1000/fps/faster))
			success, frame = capture.read()
	else:
		while success:
			frame = filter_fun(frame)
			cv2.imshow("windows", frame)
			cv2.waitKey(int(1000/fps/faster))
			success, frame = capture.read()
	capture.release()

def Video2Img(vd_path, img_path):
	#10M的视频可以拆分成接近40M的图片，重点在于图片的命名方式，因为还要重新组装成图片
	
	if not os.path.exists(img_path):
		os.mkdir(img_path)

	capture = cv2.VideoCapture(vd_path)
	
	nbframes = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
	nb = 3
	num = 0
	num_s = str(num)

	success, frame = capture.read()

	while success:
		
		while len(num_s) < 3:
			num_s = "0" + num_s

		cv2.imwrite(os.path.join(img_path, "pic_" + num_s + ".jpg"), frame)
		num += 1
		num_s = str(num)
		success, frame = capture.read()

	capture.release()
	print("Finsh cut video into several images.")

def Img2Video(img_path, vd_path):
	#将一个文件夹中的所有文件，一口气合成一个视频，至于排序，取决于os.listdir的排序方法
	
	(height, width) = cv2.imread(img_path + os.listdir(img_path)[0]).shape[:2]
	
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")

	videoWritter = cv2.VideoWriter(vd_path, fourcc, 10, (width, height))

	for img in os.listdir(img_path):
		
		im = cv2.imread(os.path.join(img_path, img))
		videoWritter.write(im)

	videoWritter.release()


if __name__ == "__main__":

	vd_dir = "D:\\0824\\Project1\\Project1\\test_videos\\"
	im_dir = "D:\\0824\\Project1\\Project1\\test_videos\\06\\"
	
	test_vd = "D:\\0824\\Project1\\Project1\\test_videos\\06.mp4"
	test_vd2 = "D:\\0824\\Project1\\Project1\\test_videos\\06_1.mp4"
	
	#Video2Img(test_vd, "D:\\0824\\Project1\\Project1\\test_videos\\" + "06")
	#Img2Video(im_dir, test_vd2)
	#
	build_test_plane()
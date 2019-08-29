import cv2
import os
import numpy as np

############ pixel-->world ##############
#逆透视变换
def pixel2world(u, v, zw = 0):
	'''
	输入的u,v是照片中的像素位置，返回的是空间中的位置关系，即世界坐标系的坐标，
	需要补充一个坐标来协助，此处我们给出的是高度zw = 0.
	'''

	#相机高度
	h = 1.8
	#焦距
	f = 0.05
	#光心位置
	opticalCenterX = 640
	opticalCenterY = 360
	#像素在图片中的尺寸
	dX = 0.0072
	dY = 0.0072

	pixel2picture = np.float32(
		[	[dX, 0, -opticalCenterX*dX],
			[0, dY, -opticalCenterY*dY],
			[0, 0, 1]	])
	
	input = np.float32([[u, v, 1]]).transpose()
	#先从像素坐标系变换到图像坐标系
	output = np.matmul(pixel2picture, input)
	#print(output)
	#从中读出X和Y
	X, Y, _ = output
	#世界坐标系中的zw已知，那么也就可以知道相机坐标系中的yc
	#再使用直线方程配合过原点的条件来推导，就可以知道另外两个轴上的坐标
	yc = h - zw
	xc = X*yc/Y
	zc = f*yc/Y

	#从相机坐标系转化成世界坐标系
	camera2world = np.float32(
		[	[1, 0, 0, 0],
			[0, 0, 1, 0],
			[0, -1, 0, h],
			[0, 0, 0, 1]])
	Pc = np.float32([[xc, yc, zc, 1]]).transpose()

	result = np.matmul(camera2world, Pc)

	return result

############ pixel-->world ##############
#透视变换
def world2pixel(xw, yw, zw, round = True):
	'''
	输入的是一个在世界坐标系中的三维坐标，计算后得到在像素上的位置
	可以通过参数round来判断是否对结果进行取整，不取整的话是为了方便后续的插值
	'''

	#相机高度
	h = 1.8
	#焦距
	f = 0.006
	#光心位置
	opticalCenterX = 640
	opticalCenterY = 360
	#像素在图片中的尺寸
	dX = 0.0072
	dY = 0.0072

	#从世界坐标系转化成相机坐标系
	world2camera = np.float32(
		[	[1, 0, 0, 0],
			[0, 0, -1, h],
			[0, 1, 0, 0],
			[0, 0, 0, 1]	])
	#从相机坐标系转化成图像坐标系，三维空间投影到二维空间上
	#s是物体在相机坐标系中的z坐标，看成一个缩放的因子
	s = yw
	if s < f:
		raise ValueError("The point is too close to the camera.")
	camera2picture = np.float32(
		[	[f, 0, 0, 0],
			[0, f, 0, 0],
			[0, 0, 1, 0]	])/s
	picture2pixel = np.float32(
		[	[1/dX, 0, opticalCenterX],
			[0, 1/dY, opticalCenterY],
			[0, 0, 1]	])

	M = np.matmul(np.matmul(picture2pixel, camera2picture), world2camera)
	#print(M, M.shape)

	input = np.float32([[xw, yw, zw, 1]]).transpose()
	#print(input, input.shape)
	#input = input.transpose()
	#print(input, input.shape)
	res = np.matmul(M, input)
	#print(res)
	#print(res, res.shape)
	if round:
		return int(res[0]), int(res[1])
	else:
		return res[0][0], res[1][0]

def world2pixel_v2(xw, yw, zw, round = True):
	'''
	输入的是一个在世界坐标系中的三维坐标，计算后得到在像素上的位置
	可以通过参数round来判断是否对结果进行取整，不取整的话是为了方便后续的插值
	用我自己的手机拍照，matlab标定得到内参矩阵，然后来计算
	'''

	#相机高度
	h = 1.7
	#焦距
	f = 0.006
	#光心位置
	opticalCenterX = 640
	opticalCenterY = 360
	#像素在图片中的尺寸
	dX = 0.0072
	dY = 0.0072

	#从世界坐标系转化成相机坐标系
	world2camera = np.float32(
		[	[1, 0, 0, 0],
			[0, 0, -1, h],
			[0, 1, 0, 0],
			[0, 0, 0, 1]	])
	#从相机坐标系转化成图像坐标系，三维空间投影到二维空间上
	#s是物体在相机坐标系中的z坐标，看成一个缩放的因子
	s = yw
	if s < f:
		raise ValueError("The point is too close to the camera.")
	intrinM = np.float32(
		[	[1.519, 0, 0.52],
			[0, 1.5142, 0.7229],
			[0, 0, 0.001],	])*1000

	M = np.matmul(intrinM, world2camera)
	#print(M, M.shape)

	input = np.float32([[xw, yw, zw, 1]]).transpose()
	#print(input, input.shape)
	#input = input.transpose()
	#print(input, input.shape)
	res = np.matmul(M, input)
	#print(res)
	#print(res, res.shape)
	if round:
		return int(res[0]), int(res[1])
	else:
		return res[0][0], res[1][0]


def plane2img():
	'''
	测试，给出空间中一个平面上的两条平行直线作为车道线，观察透视到图片上的情况
	目的是为了检查我们写透视变换的函数是否正确，如果这个函数写错了那么逆透视的函数必然是错误的
	从输出的图像中明显可以看到消失点
	'''
	x_left = [-2, -1.5]
	x_right = [1.5, 2]

	height = 72*10
	width = 128*10
	#需要切记，用nump.ndarray来表示像素时，(height, width)在我们印象中就是一个矩形，第0维是高，第1维是宽
	#在赋值的时候不能给错了
	pic = np.zeros((height, width), dtype = "uint8")

	for xw in np.linspace(-4, 4, 100):
		for yw in np.linspace(0.00, 5, 5000):
			#print(round(i, 1), round(j, 1))
			if (xw >= x_left[0] and xw <= x_left[1]) or (xw >= x_right[0] and xw <= x_right[1]):
				
				try:
					u, v = world2pixel_v2(xw, yw, 0, True)
				except ValueError:
					continue
				if (u >=0 and u< width) and (v >= 0 and v < height):

					pic[v, u] = 255

	cv2.imshow("w", pic)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite("1.jpg", pic)

def getbirdeye(src, image_h = 640, image_w = 640, ratio = 0.02, interpolation = "BILINEAR"):
	'''

	'''

	im = cv2.imread(src)
	para = im.shape
	H = para[0]
	W = para[1]
	#print(H, W)

	channel = len(para)

	if channel == 2:
		print("GRAY")
		pic = np.zeros((image_h, image_w))
	else:
		print("BGR.")
		pic = np.ones((image_h, image_w, channel))*120

	exception_num1 = 0
	exception_num2 = 0
	if interpolation == "NEAREST":
		for vw in range(image_h):
			for uw in range(image_w):
				xw = uw - image_w/2
				yw = image_h - vw
				#print(xw, yw)
				try:
					u, v = world2pixel_v2(xw*ratio, yw*ratio, 0, False)
				except ValueError:
					pic[vw, uw, :] = 50
					exception_num1 += 1
					continue
				u = int(round(u))
				v = int(round(v))
				#print(u, v)
				try:
					pic[vw, uw, :] = im[v, u, :]
				except IndexError:
					exception_num2 += 1
					continue
	elif interpolation == "BILINEAR":
		for vw in range(image_h):
			for uw in range(image_w):
				xw = uw - image_w/2
				yw = image_h - vw
				try:
					u, v = world2pixel((uw - image_w/2)*ratio, (image_h - vw)*ratio, 0)
				except ValueError:
					continue
				if u >= W or u < 0 or v >= H or v < 0:
					continue
				if u%1 != 0:
					if v%1 != 0:
						break

	cv2.imshow("BIRDEYE", pic)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	'''
	pic = cv2.GaussianBlur(pic, (3,3), 5)
	cv2.imshow("BIRDEYE", pic)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''

	print("exception_num1 is ", exception_num1)
	print("exception_num2 is ", exception_num2)

	return None

def bilinear_interpolation(x, y, src):

	if x%1 == 0:
		if y%1 == 0:
			return 

def nearest_interpolation(x, y, src):
	pass

if __name__ == "__main__":

	test_img = "D:\\0824\\Project1\\Project1\\test_videos\\pic_001.jpg"
	test1 = "D:\\GitFile\\roadlane\\1.jpg"
	test2 = "D:\\0824\\Project1\\Project1\\test_videos\\2.jpg"
	#pic = cv2.imread(test_img)

	#pixel2world(pic)
	#x, y = world2pixel(2.3, 7.8 ,0, False)
	#print(x, y)
	#print(pixel2world_v2(x, y))
	plane2img()
	#
	#getbirdeye(test1, interpolation = "NEAREST")
	#getbirdeye(test2, interpolation = "NEAREST")
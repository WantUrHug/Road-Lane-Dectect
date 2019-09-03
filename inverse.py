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
#逆透视变换2
def pixel2world_v2(u, v):
	pass

############ world-->pixel ##############
#逆透视变换3
def pixel2world_v3(u, v):
	'''
	想要多模拟一个真实情况中的相机结果，也就是建模，透视变换和逆透视变换，
	本质上也就是建模。但是似乎，有个与相机镜头息息相关的量被忽略了，孔径角，
	在一些更加复杂的逆透视变换推导中有涉及，它影响了我们能够塞进相机中的内容，
	尤其是我们的情景很大可能是朝着地面，所以是指定了一片区域，假定相机完完全全就是拍着地面
	这也可以帮助我们在建模完成后如何在使用中指导较优的选择.
	与原先版本的差距在于，暂时缺少了最后一步的，变换到真实的像素坐标系中，以左上角为原点
	而不是以中点为原点.所以在真的画图时，还需要平移.
	与world2pixel_v3对应.
	'''

	#需要的参数
	#相机的高度、焦距、分辨率、半视场角、光心位置
	f = 0.006#6mm镜头
	h = 1
	m = 1280
	n = 720
	alpha = 60/180*np.pi
	beta = 60/180*np.pi
	centerX = 640
	centerY = 360
	#偏转角和俯仰角，偏向角为0，俯仰角为90表示相机的镜头是朝着车辆的正前方
	#若偏向角为0、俯仰角为60表示稍微向下
	pitch = 0/180*np.pi
	yaw = 50/180*np.pi
	
	#相机平面上的相元尺寸，分别是u和v两个方向，利用相机的参数来进行计算
	#也就是每个像素真实的长度(?).在最后的计算中会用到但是这里是代码所
	#以没有写进去，太冗余
	#lu = 2*f*np.tan(alpha)/(m - 1)
	#lv = 2*f*np.tan(beta)/(n - 1)

	#u, v = 360 - v, u - 640

	Yw = h*np.tan(yaw - np.arctan(2*u/(m - 1)*np.tan(alpha)))
	Xw = np.sqrt(h**2 + Yw**2)*2*v/(n - 1)*np.tan(beta)/np.sqrt(1 + (2*u/(m - 1)*np.tan(alpha))**2)

	return Xw, Yw

############ world-->pixel ##############
#透视变换
def world2pixel(xw, yw, zw, round = True):
	'''
	最理想的透视变换.
	输入的是一个在世界坐标系中的三维坐标，计算后得到在像素上的位置.
	可以通过参数round来判断是否对结果进行取整，不取整的话是为了方便后续的插值.
	不考虑视场角和分辨率等参数.
	'''

	#相机高度
	h = 1.8
	#焦距
	f = 0.006
	#光心位置
	opticalCenterX = 640
	opticalCenterY = 360
	#像素在图片坐标系中的尺寸
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

############ world-->pixel ##############
#透视变换2
def world2pixel_v2(xw, yw, zw, round = True):
	'''
	输入的是一个在世界坐标系中的三维坐标，计算后得到在像素上的位置。
	可以通过参数round来判断是否对结果进行取整，不取整的话是为了方便后续的插值。
	用自己的手机拍照，matlab标定得到内参矩阵，然后来计算。
	优点在于避免了相机内在的缺陷，但是缺点在于距离稍远会由于插值导致逆透视变换的效果很差，
	并且没有完全利用ROI中的点
	'''

	#相机高度
	h = 0.357
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
		[	[1.2663, 0, 0.5435, 0],
			[0, 1.2659, 0.7256, 0],
			[0, 0, 0.001, 0],	])*1000
	#print("...")
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

def world2pixel_v3(xw, yw, zw = 0):
	'''
	建立透视变换.和之前的版本有不少区别，例如光心居于图像和实物中间，让我们可以不担心太近的点越过边界.
	引入视场角!
	'''


	#需要的参数
	#相机的高度、焦距、分辨率、半视场角、光心
	f = 0.006#6mm镜头
	h = 1
	m = 1280
	n = 720
	alpha = 60/180*np.pi
	beta = 60/180*np.pi
	centerX = 640
	centerY = 360
	#偏转角和俯仰角，偏向角为0，俯仰角为90表示相机的镜头是朝着车辆的正前方
	#若偏向角为0、俯仰角为60表示稍微向下
	pitch = 0
	yaw = 60/180*np.pi

	u = (m-1)/2/np.tan(alpha)*np.tan(yaw - np.arctan(yw/h))
	v = np.sqrt(1 + (2*u/(m - 1) * np.tan(alpha))**2) * xw/np.sqrt(h**2 + yw**2)*(n - 1)/2/np.tan(beta)

	return u, v


def plane2img(src = None):
	'''
	测试，给出空间中一个平面上的两条平行直线作为车道线，观察透视到图片上的情况
	目的是为了检查我们写透视变换的函数是否正确，如果这个函数写错了那么逆透视的函数必然是错误的
	从输出的图像中明显可以看到消失点
	'''

	#两条白色线所在的位置确定下来
	x_left = [-2, -1.5]
	x_right = [1.5, 2]

	height = 72*10
	width = 128*10
	#需要切记，用nump.ndarray来表示像素时，(height, width)在我们印象中就是一个矩形，第0维是高，第1维是宽
	#在赋值时要注意 u 和 v 代表的含义，不能给错了
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
def plane2img_v2(src, H = 640, W = 960, world2pixel = False, interpolation = "NEAREST"):
	'''
	与版本一不同，是将我原本绘制好的平面图片，当成空间中的平面，处理成相片的内容.
	参数world2pixel决定了，在绘制相片时，是采用透视变换还是逆透视变换，False表示
	pixel2world，即逆透视变换的觅值方法.
	'''
	imarr = np.zeros((H, W), dtype = "uint8")
	origin = cv2.imread(src, 0)
	print(origin.shape)
	origin_h, origin_w = origin.shape

	cv2.imshow("A", origin)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	ratio = 0.01

	if not world2pixel:
		if interpolation == "NEAREST":
			for v in range(H):
				for u in range(W):
					#平移到以图片中心为原点的坐标系中
					uu = 480 - v
					vv = u - 320
	
					x, y = pixel2world_v3(uu, vv)
					#print(x, y)
					x /= ratio
					y /= ratio
					x += origin_w/2
					#y += origin_h/2
					if (x < - 1/2 or x >= origin_w - 1/2) or (y < - 1/2 or y >= origin_w - 1/2):
						#逆透视之后的点不在平面的范围之中，算了算了
						break
					x = int(round(x))
					y = int(round(y))
					imarr[v, u] = origin[x][y]

			cv2.imshow("B", imarr)
			cv2.waitKey(0)
			cv2.destroyAllWindows()










def getbirdeye(src, image_h = 640, image_w = 480, ratio = 0.005, interpolation = "BILINEAR"):
	'''

	'''

	#im = cv2.imread(src)
	im = src
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
		pic = np.zeros((image_h, image_w, channel))

	exception_num1 = 0
	exception_num2 = 0
	if interpolation == "NEAREST":
		print("choose nearest interpolation")
		for vw in range(image_h):
			for uw in range(image_w):
				xw = uw - image_w/2
				yw = image_h - vw
				#print(xw, yw)
				try:
					u, v = world2pixel_v2(xw*ratio, yw*ratio, 0, False)
				except ValueError:
					
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
	test3 = "D:\\inverse_inspective\\3.jpg"
	#pic = cv2.imread(test_img)

	#pixel2world(pic)
	#x, y = world2pixel(2.3, 7.8 ,0, False)
	#print(x, y)
	#print(pixel2world_v2(x, y))
	#plane2img_v2(test1)
	#getbirdeye(test3, interpolation = "NEAREST")
	#getbirdeye(test2, interpolation = "NEAREST")
	u, v = world2pixel_v3(-0.2, 0.5)
	print(u, v)
	x, y = pixel2world_v3(u, v)
	print(x, y)
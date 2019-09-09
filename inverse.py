import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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

############ world-->pixel ##############
#逆透视变换3
def pixel2world_v3(u, v):
	'''
	想要多模拟一个真实情况中的相机结果，也就是建模，透视变换和逆透视变换，
	本质上也就是建模。但是似乎，有个与相机镜头息息相关的量被忽略了，孔径角，
	在一些更加复杂的逆透视变换推导中有涉及，它影响了我们能够塞进相机中的内容，
	尤其是我们的情景很大可能是朝着地面，所以是指定了一片区域，假定相机完完全全就是拍着地面，
	所以要进一步约束视场角和俯仰角之间的关系.
	这也可以帮助我们在建模完成后如何在使用中指导较优的选择.
	与原先版本的差距在于，暂时缺少了最后一步的，变换到真实的像素坐标系中，以左上角为原点
	而不是以中点为原点.所以在真的画图时，还需要平移.
	与world2pixel_v3对应.
	'''

	#需要的参数
	#相机的高度、焦距、分辨率、半视场角
	f = 0.006#6mm镜头
	h = 1
	m = 640
	n = 480
	alpha = 30/180*np.pi
	beta = 30/180*np.pi
	#理想情况下光心应该就在图像的中心，光心实际上起的作用不大，在这里是为了实现从
	centerX = 640/2
	centerY = 360/2
	
	#偏转角和俯仰角，偏向角为0，俯仰角为90表示相机的镜头是朝着车辆的正前方
	#若偏向角为0、俯仰角为60表示稍微向下
	pitch = 0/180*np.pi
	yaw = 60/180*np.pi
	
	#相机平面上的相元尺寸，分别是u和v两个方向，利用相机的参数来进行计算
	#也就是每个像素真实的长度(?).在最后的计算中会用到但是这里是代码所
	#以没有写进去，太冗余
	#lu = 2*f*np.tan(alpha)/(m - 1)
	#lv = 2*f*np.tan(beta)/(n - 1)

	#u, v = 360 - v, u - 640

	Yw = h*np.tan(yaw - np.arctan(2*u/(m - 1)*np.tan(alpha)))
	Xw = np.sqrt(h**2 + Yw**2)*2*v/(n - 1)*np.tan(beta)/np.sqrt(1 + (2*u/(m - 1)*np.tan(alpha))**2)

	return Xw, Yw

def pixel2world_v3_1(u, v, YAW, h, n, m):
	'''
	基本的逻辑同上个函数，只是一些参数要采用现实的手机的结果.
	'''

	#手动测量的视场角，要除以2，角度其实真的很小
	alpha = 45/2/180*np.pi
	beta = 45/2/180*np.pi
	
	#偏转角和俯仰角，偏向角为0，俯仰角为90表示相机的镜头是朝着车辆的正前方
	#若偏向角为0、俯仰角为60表示稍微向下
	pitch = 0/180*np.pi
	yaw = YAW/180*np.pi
	
	#相机平面上的相元尺寸，分别是u和v两个方向，利用相机的参数来进行计算
	#也就是每个像素真实的长度(?).在最后的计算中会用到但是这里是代码所
	#以没有写进去，太冗余

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
	#f = 0.006#6mm镜头
	h = 1
	m = 640
	n = 480
	alpha = 30/180*np.pi
	beta = 30/180*np.pi
	centerX = m/2
	centerY = n/2
	#偏转角和俯仰角，偏向角为0，俯仰角为90表示相机的镜头是朝着车辆的正前方
	#若偏向角为0、俯仰角为60表示稍微向下
	pitch = 0
	yaw = 60/180*np.pi

	u = (m-1)/2/np.tan(alpha)*np.tan(yaw - np.arctan(yw/h))
	v = np.sqrt(1 + (2*u/(m - 1) * np.tan(alpha))**2) * xw/np.sqrt(h**2 + yw**2)*(n - 1)/2/np.tan(beta)

	return u, v

def world2pixel_v3_1(xw, yw, YAW, h, m, n):
	'''
	和上面的函数
	'''

	#手动测量的视场角，要除以2，角度其实真的很小
	alpha = 45/2/180*np.pi
	beta = 46/2/180*np.pi

	#偏转角和俯仰角，偏向角为0，俯仰角为90表示相机的镜头是朝着车辆的正前方
	#若偏向角为0、俯仰角为60表示稍微向下
	pitch = 0
	yaw = YAW/180*np.pi

	u = (m-1)/2/np.tan(alpha)*np.tan(yaw - np.arctan(yw/h))
	v = np.sqrt(1 + (2*u/(m - 1) * np.tan(alpha))**2) * xw/np.sqrt(h**2 + yw**2)*(n - 1)/2/np.tan(beta)

	return -u, -v

def plane2img(src = None):
	'''
	测试，给出空间中一个平面上的两条平行直线作为车道线，观察透视到图片上的情况
	目的是为了检查我们写透视变换的函数是否正确，如果这个函数写错了那么逆透视的函数必然是错误的
	从输出的图像中明显可以看到消失点
	'''

	imarr = np.zeros((H, W), dtype = "uint8")

	if not world2pixel:
		for v in range(H):
			for u in range(W):
				#平移到以图片中心为原点的坐标系中
				uu = H/2 - v
				vv = u - W/2

				x, y = pixel2world_v3(uu, vv)
				#print(x, y)
				imarr[v, u] = _plane_color(x, y)

		cv2.imshow("Pixel", imarr)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def plane2img_v2(src, H = 480, W = 640, world2pixel = False):
	'''
	与版本一不同，是将我原本绘制好的平面图片，当成空间中的平面，处理成相片的内容.
	参数world2pixel决定了，在绘制相片时，是采用透视变换还是逆透视变换，False表示
	pixel2world，即逆透视变换的觅值方法.
	但有个问题，平面图片原本就是像素点，似乎不太好?!应该要建立起空间中更加细微的关系，
	例如空间中x坐标在(0.5, 0.8)这个范围内的点就是白色的.所以要额外设置一个函数，用来当给定一个点时，准确的
	返回这个点的颜色情况，而不是还要在一张离散的图片中去，这样的插值没有太大意义
	'''
	imarr = np.zeros((H, W), dtype = "uint8")

	if not world2pixel:
		for v in range(H):
			for u in range(W):
				#平移到以图片中心为原点的坐标系中
				uu = H/2 - v
				vv = u - W/2

				x, y = pixel2world_v3(uu, vv)
				#print(x, y)
				imarr[v, u] = _plane_color(x, y)

		cv2.imshow("Pixel", imarr)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	getbirdeye_v3(imarr)

def plane2img_v3(H, W):
	'''
	测试，想做一个动图来观察一个代表车子的矩形框在车道中旋转(实际上是车道在旋转，
	只是效果相同).物体和直线比较多所以不再使用黑白图像了，用彩色的RGB，更清楚一点
	'''

	imarr = np.zeros((H, W, 3), dtype = "uint8")

	for v in range(H):
		for u in range(W):
			#平移到以图片中心为原点的坐标系中
			uu = H/2 - v
			vv = u - W/2
			x, y = pixel2world_v3(uu, vv)
			#print(x, y)
			imarr[-v, u] = _plane_color_v3(x, y, 80)

	cv2.imshow("Pixel", imarr)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def _plane_color(x, y):
	'''
	给入一个点的坐标，如果符合车道线的所在位置，就返回255表示白色.
	如果不是车道线就返回0.此处使用连续的车道线，后期可以尝试间断的
	车道线或者是转弯的车道线.
	如果以一条马路中间为原点，用线画一个简图如下:

	|
	|
	|||||||||||||||||||||||||||||||||||||||
	|
	|
	|
	|------------------------------------> y轴正
	|
	|
	|
	|||||||||||||||||||||||||||||||||||||||
	|
	|
	|
	V x轴正
	
	'''
	#实线宽度，20cm
	thick = 0.2
	#车道宽度
	width = 3
	absx = abs(x)
	if (absx > width/2 and absx < width/2 + thick):
		return 255
	else:
		return 0
def _plane_color_v2(x, y):
	'''
	画有一定角度的直线.angle是从x正半轴到y正半轴开始计算.
	|
	|    ||
	|  ||
	|||
	|
	|
	|------------------------------------->
	|	 ||
	|  ||
	|||
	|
	|
	'''
	thick = 0.2
	width = 4
	angle = 75/180*np.pi#angle != 0.

	cut = abs(x - y/np.tan(angle))

	if (cut < width/2 + thick/np.sin(angle) and cut > width/2):
		return 255
	else:
		return 0

def _plane_color_v3(x, y, theta = None):

	#car_l = 0.1
	car_w = 2	
	#红色线表示车子的方向.
	car_center = [0,0,255]
	#在俯视图中则使用XX色来表示车身.
	car_color = [255, 255, 0]

	road_width = 3
	road_thick = 0.2
	#马路仍使用白色表示
	road_color = [255, 255, 255]

	#tan1 = car_w/car_l
	#tan2 = (car_w**2+car_l**2-road_width**2)/road_width/np.sqrt(car_w**2+car_l**2)

	#max_k = (tan1+tan2)/(1+tan1*tan2)
	#min_kk = 1/max_k

	result = [0, 0, 0]
	theta = theta/180*np.pi

	#if np.tan(theta) < max_k:
	#	raise ValueError("theta too small!")
	if theta <= 0 and theta >= 90:
		raise ValueError("Should between 0-90 degree.")
	#限制角度有一个好处，就是在xOy坐标中，直线总是从左下到右上的
	#方向固定了之后就可以很简单的判断
	#不过似乎加上斜率也可以用一个式子解决，节省判断的开销

	k = np.tan(theta)
	b = 3 + k
	#b怎么算的？是已知k，然后点(-1,3)，也就是车的左上角在线上 3=-1*k+b => b=3+k
	#k*(kx+b-y)可以判断点在左右，因为(kx+b-y)是判断在上下的，加上斜率就可以判断左右
	#flag = (k > 0)*(k*x+b-y)
	#按照车道线的概念，现在是在考虑左侧的线，那么flag就是这条线右侧，左侧那条边线的方程位
	#也可以用相同的方法来构造，但是说实话还是比较麻烦，所以还是用之前那个观察在x轴上的截距来判断
	left_flag_right = -b/k
	left_flag_left = -b/k - road_thick/np.sin(theta)
	x_cut = x - y/k
	if x_cut < left_flag_right and x_cut > left_flag_left:
		result = road_color
	
	mid_flag_left = -b/k  + road_width/np.sin(theta)/2
	mid_flag_right = mid_flag_left + road_thick/np.sin(theta)
	if x_cut < mid_flag_right and x_cut > mid_flag_left:
		result = road_color

	right_flag_left = -b/k + road_width/np.sin(theta)
	right_flag_right = right_flag_left + road_thick/np.sin(theta)
	if x_cut < right_flag_right and x_cut > right_flag_left:
		result = road_color

	#if abs(x) < car_w/2 and (y > 0 and y < 3):
	#	result = car_color
	if abs(x) < 0.05:
		result = car_center
	return result

def getbirdeye_v3(src, image_h = 640, image_w = 960):
	'''
	利用公式(3)的逆透视变换来执行得到俯瞰图.
	为了严谨，需要考虑消失点！但是我们这个假设的前提，就是相机对着地面，所以不会有消失点.
	所以我们可以试着取图片中四条边的中点附近的点作为极限，优先计算这四点映射到平面上的位置，
	以此来确定俯视图中每个像素对应现实中尺寸的大小.
	考虑更好的双线性插值.
	'''

	if isinstance(src, np.ndarray):
		im = src
	else:
		im = cv2.imread(src)

	height, width = im.shape[:2]
	#print(height, width)


	uvlimits = np.ones((4,3))
	uvlimits[:, :2] = np.array([[1,1], [width - 2, height - 2], [width - 2, 1], [1, height - 2]])

	u_uu = np.array(
		[	[0, 1, 0],
			[-1, 0, 0],
			[(height-1)/2, -(width-1)/2, 1]	])
	uvlimits = np.matmul(uvlimits, u_uu)

	xylimits = np.array([pixel2world_v3(i[0], i[1]) for i in uvlimits])
	xMax = max(xylimits[:, 0])
	xMin = min(xylimits[:, 0])
	yMax = max(xylimits[:, 1])
	yMin = min(xylimits[:, 1])
	
	print(xMax, xMin)
	print(yMax, yMin)

	#根据计算出来的空间范围来实际的计算每个像素在不同方向上所代表的实际距离
	step_y = (yMax - yMin)/image_h
	step_x = (xMax - xMin)/image_w
	print("step x: %.4f, step y: %.4f"%(step_x, step_y))
	result = np.zeros((image_h, image_w))

	y = yMin
	for i in range(image_h):
		x = xMin
		for j in range(image_w):
			u, v = world2pixel_v3(x, y)
			uu, vv = v + width/2, height/2 - u
			if (uu < 1 or uu >= width-1) or (vv < 1 or vv >= height - 1):
				result[i, j] = 0
				x += step_x
				continue
			u1, u2 = int(uu), int(uu + 1)
			v1, v2 = int(vv), int(vv + 1)
			delta_u = uu - u1
			delta_v = vv - v1
			
			val = im[v1, u1]*(1-delta_u)*(1-delta_v)+im[v1, u2]*delta_u*(1-delta_v)+im[v2, u2]*delta_u*delta_v+im[v2, u1]*(1-delta_u)*delta_v
			result[i, j] = val/4
			x += step_x
		y += step_y
	print(x, y)

	cv2.imshow("BIRD", result)
	cv2.waitKey(1000)
	#cv2.destroyAllWindows()
	cv2.imwrite("Bird.jpg", result)

	analyse_birdeye_v2(result, step_x, step_y)

	return result, (step_x, step_y)

def getbirdeye_v3_1(src, yaw, h, image_h = 640, image_w = 960):
	'''
	特别的来处理陈俊他的手机拍摄的照片.
	需要做一些视角转换的工作，因为拍摄的照片都是自下而上，平行直线收敛
	符合人的认知习惯。之前推导模型的时候由于光心位于物体和成像中间，所以
	会出现倒转的现象，现在要把 world2pixel_v3_1 和 pixel2world_v3_1
	都适当修改，需要可以适应不同尺寸的照片，无需再手动去改参数
	'''

	im = cv2.imread(src, 0)

	#查看尺寸
	height, width = im.shape[:2]
	print(height, width)

	#相机标定的结果
	matlab_result = np.array([ 	[3.8465, 0, 0],
						[0, 3.8495, 0],
						[1.6925, 1.7475, 0.001]		])*1000
	
	#principleX principleY光心位置
	#光心的位置是根据标定的结果，按比例换算的，原始的相机像素似乎就是 3456x3456，按比例换算过来之后
	#没问题
	M = 3456
	N = 3456
	pX = matlab_result[2,0]*width/M
	pY = matlab_result[2,1]*height/N
	print("Principle: ", pX, pY)

	uvlimits = np.ones((4,3))
	uvlimits[:, :2] = np.array([[1,1], [width - 2, height - 2], [width - 2, 1], [1, height - 2]])

	u_uu = np.array(
		[	[0, 1, 0],
			[-1, 0, 0],
			[pY, -pX, 1]	])
	uvlimits = np.matmul(uvlimits, u_uu)

	xylimits = np.array([pixel2world_v3_1(i[0], i[1], yaw, h, height, width) for i in uvlimits])
	xMax = max(xylimits[:, 0])
	xMin = min(xylimits[:, 0])
	yMax = max(xylimits[:, 1])
	yMin = min(xylimits[:, 1])
	
	print(xMax, xMin)
	print(yMax, yMin)

	#根据计算出来的空间范围来实际的计算每个像素在不同方向上所代表的实际距离
	step_y = (yMax - yMin)/image_h
	step_x = (xMax - xMin)/image_w
	print("step x: %.5f, step y: %.5f"%(step_x, step_y))
	result = np.zeros((image_h, image_w), dtype = "uint8")

	y = yMin
	for i in range(image_h):
		x = xMin
		for j in range(image_w):
			u, v = world2pixel_v3_1(x, y, yaw, h, height, width)
			uu, vv = v + pX, pY - u
			if (uu < 1 or uu >= width-1) or (vv < 1 or vv >= height - 1):
				result[i, j] = 0
				x += step_x
				continue
			u1, u2 = int(uu), int(uu + 1)
			v1, v2 = int(vv), int(vv + 1)
			delta_u = uu - u1
			delta_v = vv - v1
			
			val = im[v1, u1]*(1-delta_u)*(1-delta_v)+im[v1, u2]*delta_u*(1-delta_v)+im[v2, u2]*delta_u*delta_v+im[v2, u1]*(1-delta_u)*delta_v
			if val > 255:
				val = 255
			result[-i, j] = val
			x += step_x
		y += step_y
	print(x, y)

	cv2.imshow("BIRD%d"%yaw, result)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	#cv2.imwrite("B%d_%.2f.jpg"%(yaw,h), result)

def analyse_birdeye(result, step_x, step_y):
	#需要知道图像中像素，在横纵方向上各代表多少距离
	#从图像中我们也可以获知，第二由白转黑和第二个由黑转白的点就是车道内侧的位置，
	#结合距离，看看距离车道线宽度的误差变化
	#这个函数适用于已知图像中的线和镜头方向平行，即 	_plane_color 这个函数
	
	#cv2.imshow("BIRD", result)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	image_h, image_w = result.shape
	#print(result)
	record = []

	for i in range(image_h - 1):
		white2black = 0
		black2white = 0
		left = 0
		right = 0
		for j in range(image_w - 1):
			if result[i, j] == 0 and result[i, j + 1] != 0:
				black2white += 1
			if result[i, j] != 0 and result[i, j + 1] == 0:
				white2black += 1
			if  white2black == 1:
				left = j
			if black2white == 2:
				right = j
				break
		record.append((right - left)*step_x)

	row = range(len(record))
	plt.plot(row, record, label = "Predict")
	plt.plot(row, [2/np.sin(75/180*np.pi) for i in row], label = "Reality")
	plt.legend()
	plt.show()

def analyse_birdeye_v2(result, step_x, step_y):
	#获得的图像可以看成是已经是经过了二值化处理的，但是边界两侧的区域确实有点难处理，只能在画俯视图时
	#把扇形区域以外的地方也填充成黑色
	#然后图像中如果顺利，会有两条白色的车道线，但也有可能只有一条，所以最好做一个聚类分析？
	#太麻烦了，要不就直接做判断，把所有点集合起来做一个线性回归拟合，得到一条直线，如果点离直线都很近(0.2)就
	#确定是一条，否则就是两条，就把点在这条直线左右分成两组，再各自去求拟合.如果继续递归，那么还可以
	#区分出更多的等距平行线，关键是平行.如果利用更复杂的聚类算法就可以不受平行这个条件的约束.
	#目前是只有一或二，所以不用写成递归.
	
	image_h, image_w = result.shape

	record = []

	for i in range(200, image_h - 1):
		#white2black = 0
		#black2white = 0
		exchange = 0
		left = 0
		right = 0
		for j in range(image_w - 1):
			if (result[i, j] == 0 and result[i, j + 1] != 0) or (result[i, j] != 0 and result[i, j + 1] == 0):
				exchange += 1
			if result[i, j] == result[i, j + 1]:
				continue
			if exchange == 2:
				left = j
			if exchange == 3:
				right = j
				break
		record.append((right - left)*step_x)

	row = range(len(record))
	plt.plot(row, record, label = "Predict")
	plt.plot(row, [3 for i in row], label = "Reality")
	plt.legend()
	plt.show()

if __name__ == "__main__":

	test_img = "D:\\0824\\Project1\\Project1\\test_videos\\pic_001.jpg"
	test1 = "D:\\GitFile\\roadlane\\Images\\test02.jpg"
	test2 = "D:\\0824\\Project1\\Project1\\test_videos\\2.jpg"
	test3 = "D:\\GitFile\\roadlane\\Images\\test01.jpg"
	#pic = cv2.imread(test_img)

	#pixel2world(pic)
	#x, y = world2pixel(2.3, 7.8 ,0, False)
	#print(x, y)
	#print(pixel2world_v2(x, y))
	#plane2img_v2(test1)
	#plane2img_v2(test3)
	#getbirdeye(test3, interpolation = "NEAREST")
	getbirdeye_v3_1(test1, 45, 0.328)
	#getbirdeye_v3_1(test3, 61, 1.28)
	#u, v = world2pixel_v3(-0.2, 2)
	#print(u, v)
	#x, y = pixel2world_v3(u, v)
	#print(x, y)
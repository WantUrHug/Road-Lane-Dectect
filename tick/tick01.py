import cv2
import numpy as np
from numpy import tan, cos, sin

theta = 70*np.pi/180
alpha = 45/2*np.pi/180
m = 3456
n = 3456

matlab_result = np.array(
	[	[3.8465, 0, 0],
    	[0, 3.8495, 0],
    	[1.6925, 1.7475, 0.001]	])*1000

focalX = matlab_result[2, 0]
focalY = matlab_result[2, 1]

im = cv2.imread("tick_test02.jpg")

im = cv2.resize(im, (0, 0), fx = 0.2, fy = 0.2, interpolation=cv2.INTER_CUBIC)
edge = cv2.Canny(im, 50, 180, 0)
#cv2.imwrite("edge1.jpg", edge)
#cv2.namedWindow("EDGE", cv2.WINDOW_NORMAL)

kernel = np.ones((3,3), np.uint8)
#erosion = cv2.erode(edge, kernel, iterations=1)
dilate = cv2.dilate(edge, kernel, iterations=1)

#cv2.imshow("EDGE", dilate)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

lines = cv2.HoughLinesP(dilate, 1, np.pi/180, 31, 70, 60)
print("The number of lines detected by HoughTransform: %d"%len(lines))

h, w, _ = im.shape
line = lines[:, 0, :]
filter_line = []
for x1, y1, x2, y2 in line:
	if x1 == x2:
		continue
	else:
		k = abs((y2 - y1)/(x2 - x1))
		if k <= tan(30*np.pi/180) and k >= tan(20*np.pi/180):
			filter_line.append([x1, y1, x2, y2])
print("The number of filter lines detected by HoughTransform: %d"%len(filter_line))

#draw out all filter line.
#for x1, y1, x2, y2 in line:
#	cv2.line(im, (x1, y1), (x2, y2), (0,0,255), 5)

#draw out all filter line.
for x1, y1, x2, y2 in filter_line:
	cv2.line(im, (x1, y1), (x2, y2), (0,0,255), 2)

#cv2.imshow("Filter line", im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

centerXs = []
centerYs = []
ks = []
for x1, y1, x2, y2 in filter_line:
	centerXs.append((x1 + x2)/2)
	centerYs.append((y1 + y2)/2)
	ks.append((y2 - y1)/(x2 - x1))
print(centerXs)
print(centerYs)
print(ks)

left = -1
right = -1
flag = 0
for i, (x1, y1, x2, y2) in enumerate(filter_line):
	#这个地方有点无可奈何，因为肉眼观察到的，边缘又又又找歪了...
	#应该是右边那堆线的最左一条，左边的本应同理，不过恰好左边只有一条，
	#所以也就不用特意去写判断，以后如果有落地的空间就要考虑
	
	#总的来说，就是找出最靠左的两条，或者是，离中线最近的两条？

	if centerXs[i] == min(centerXs):
		if left < 0:
			left = i
			#centerXs.pop(0)
			cv2.line(im, (x1, y1), (x2, y2), (0, 255, 255), 2)
	else:
		if ks[i] > 0 and ks[i] == max(ks):
			right = i
			cv2.line(im, (x1, y1), (x2, y2), (255, 0, 255), 2)




#cv2.line(im, (20, 40), (h - 100, w - 200), (0, 255, 255), 5)
cv2.namedWindow("LINE", cv2.WINDOW_NORMAL)
cv2.imshow("LINE", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

new_focalX = focalX*0.2
new_focalY = focalY*0.2

dis = new_focalY*(tan(theta)-tan(alpha))/tan(alpha)
print(dis)

bottom_y = w - 1 + dis
left_x = (bottom_y - centerYs[left])/ks[left] + centerXs[left]
right_x = (bottom_y - centerYs[right])/ks[right] + centerXs[right]
print(left_x, right_x)
center2right = (right_x - new_focalX)/(right_x - left_x)
print(new_focalX)
print(center2right)
print((54.3-26.7)/54.3)

#把整张图片的延长线、虚构线和原图都画出来！
#right_x = int(right_x)#2748
#left_x = int(left_x)#-5336
#bottom_y = int(bottom_y)#1968
#new_w = right_x - left_x + 100
#new_h = bottom_y

#new_im = np.zeros((new_h, new_w, 3), dtype = "uint8")
#new_im[:h, -left_x:-left_x + w, :] = im

#cv2.line(new_im, (int(centerXs[right]- left_x), int(centerYs[right])), (new_w-100, bottom_y), (0, 0, 255), 5)
#cv2.line(new_im, (int(centerXs[left]- left_x), int(centerYs[left])), (0, bottom_y), (0, 0, 255), 5)
#cv2.line(new_im, (int(new_focalX- left_x), h), (int(-left_x + new_focalX), bottom_y), (0, 0, 255), 5)
#cv2.circle(new_im, (int(centerXs[left]- left_x), int(centerYs[left])), 10,  (0, 0, 255), 100)
#cv2.imshow("TOTAL", new_im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite("total.jpg", new_im)


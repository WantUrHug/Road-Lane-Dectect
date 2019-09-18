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

im = cv2.imread("tick_test1.jpg")
h, w, _ = im.shape
#im = im[:, :, :]

im = cv2.resize(im, (0, 0), fx = 0.2, fy = 0.2, interpolation=cv2.INTER_CUBIC)
edge = cv2.Canny(im, 150, 200, 5)
#cv2.imwrite("edge1.jpg", edge)
#cv2.namedWindow("EDGE", cv2.WINDOW_NORMAL)
#cv2.imshow("EDGE", edge)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
lines = cv2.HoughLinesP(edge, 1, np.pi/180, 50, 20, 20)
print("The number of lines detected by HoughTransform: %d"%len(lines))

h, w, _ = im.shape
line = lines[:, 0, :]
filter_line = []
for x1, y1, x2, y2 in line:
	if x1 == x2:
		#print("x1==x2")
		continue
	else:
		k = abs((y2 - y1)/(x2 - x1))
		if k <= tan(60*np.pi/180) and k >= tan(20*np.pi/180):
			filter_line.append([x1, y1, x2, y2])
print("The number of filter lines detected by HoughTransform: %d"%len(filter_line))

#draw out all filter line.
#for x1, y1, x2, y2 in line:
#	cv2.line(im, (x1, y1), (x2, y2), (0,0,255), 5)


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

for i, (x1, y1, x2, y2) in enumerate(filter_line):
#print("left lines is %d"%len(line))
	#print(centerXs[i])
	if centerXs[i] == min(centerXs):
		left = i
		cv2.line(im, (x1, y1), (x2, y2), (0, 255, 255), 5)
	if centerXs[i] == max(centerXs):
		right = i	
		cv2.line(im, (x1, y1), (x2, y2), (255, 255, 0), 5)

#cv2.line(im, (20, 40), (h - 100, w - 200), (0, 255, 255), 5)
cv2.namedWindow("LINE", cv2.WINDOW_NORMAL)
cv2.imshow("LINE", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print(im.shape)
#
new_focalX = focalX*0.2
new_focalY = focalY*0.2

dis = new_focalY*(tan(theta)-tan(alpha))/tan(alpha)
print(dis)
#cv2.line(im, (0, h + int(dis)), (w - 1, h + int(dis)), (0, 255, 255), 5)
#cv2.imshow("R", im)
#cv2.waitKey(0)
#cv2.imwrite( "with_line.jpg", im)
#cv2.destroyAllWindows()

#print(left, right)
bottom_y = w - 1 + dis
left_x = (bottom_y - centerYs[left])/ks[left] + centerXs[left]
right_x = (bottom_y - centerYs[right])/ks[right] + centerXs[right]
print(left_x, right_x)
center2right = (right_x - new_focalX)/(right_x - left_x)
print(new_focalX)
print(center2right)
print((62.1-37.5)/62.1)

#把整张图片的延长线、虚构线和原图都画出来！
right_x = int(right_x)#2748
left_x = int(left_x)#-5336
bottom_y = int(bottom_y)#1968
new_w = right_x - left_x + 100
new_h = bottom_y

new_im = np.zeros((new_h, new_w, 3), dtype = "uint8")
new_im[:h, -left_x:-left_x + w, :] = im

cv2.line(new_im, (int(centerXs[right]- left_x), int(centerYs[right])), (new_w-100, bottom_y), (0, 0, 255), 5)
cv2.line(new_im, (int(centerXs[left]- left_x), int(centerYs[left])), (0, bottom_y), (0, 0, 255), 5)
cv2.line(new_im, (int(new_focalX- left_x), h), (int(-left_x + new_focalX), bottom_y), (0, 0, 255), 5)
#cv2.circle(new_im, (int(centerXs[left]- left_x), int(centerYs[left])), 10,  (0, 0, 255), 100)
#cv2.imshow("TOTAL", new_im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("total.jpg", new_im)


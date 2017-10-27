import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

	# Take each frame
	_, frame = cap.read()

	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# define range of blue color in HSV
	lower_blue = np.array([75,120,0])
	upper_blue = np.array([90,255,255])

	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv, lower_blue, upper_blue)

	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(frame,frame, mask= mask)

	kernel = np.ones((3,3),np.uint8)
	
	erosion = cv2.erode(res,kernel,iterations = 1)

	dilation = cv2.dilate(erosion,kernel,iterations = 1)

	cv2.imshow('frame',frame)
	cv2.imshow('mask',mask)
	cv2.imshow('res',dilation)
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
	    break

cv2.destroyAllWindows()
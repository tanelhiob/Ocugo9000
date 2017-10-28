import cv2
import numpy as np

#ORANGE 22
#YELLOW 38
#GREEN 75
#BLUE 130
#VIOLET 160
#RED 179

kernel = np.ones((5,5),np.uint8)
lower = np.array([50,50,0])
upper = np.array([95,255,255])
doProcessing = True


lower2 = np.array([0,0,5])
upper2 = np.array([179,255,255])
shape = cv2.imread('./sabloon.jpg')
hsv = cv2.cvtColor(shape, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower2, upper2)
res2 = cv2.bitwise_and(shape,shape, mask= mask)

res2 = cv2.erode(res2,kernel,iterations = 1)
res2 = cv2.dilate(res2,kernel,iterations = 1)
shapeGray = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
#thresh = cv2.threshold(shapeGray,127,255,cv2.THRESH_BINARY)
_, cntCompare, _ = cv2.findContours(shapeGray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cap = cv2.VideoCapture(0)

def getGreenLight (cap):
	
	# Take each frame
	_, frame = cap.read()

	#frame = cv2.imread('./foor-4.jpeg')

	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# define range of blue color in HSV
	# hue, sat, bright
	#lower = np.array([70,85,50])
	#upper = np.array([95,255,255])
	lower = np.array([70,150,50])
	upper = np.array([95,255,255])

	# Threshold the HSV image to get only specified colors
	mask = cv2.inRange(hsv, lower, upper)

	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(frame,frame, mask= mask)

	kernel2 = np.ones((2,2),np.uint8)
	res = cv2.erode(res,kernel2,iterations = 1)
	res = cv2.dilate(res,kernel2,iterations = 1)

	res = cv2.medianBlur(res, 1)

	grey = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
	_, cntCam, _ = cv2.findContours(grey,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	#print(len(cntCam))

	probableContours = list(filter(lambda x: x.size > 25, cntCam))

	matches = list(map(lambda x: (x, cv2.matchShapes(x, cntCompare[0], 1, 0.0)), probableContours))
	#match = cv2.matchShapes(cntCam[0],cntCompare[0],1,0.0)
	#print(matches)

	matches.sort(key = lambda x: x[1])
	

	red = (0, 0, 255)
	green = (0, 255, 0)


	#colored = cv2.circle(median,(25,25), 25, green, -1)

	#cv2.drawContours(shape,cntCompare,-1,(0,0,255),1)
	if len(matches) > 0:
		print(matches[0][1])
		cv2.drawContours(res, matches[0][0],-1,(0,0,255),1)
	cv2.drawContours(shape,cntCompare,-1,(0,0,255),1)
	#cv2.imshow('shape', shape)
	cv2.imshow('frame', frame)
	cv2.imshow('res', res)

	#cv2.imshow('frame',frame)
	#cv2.imshow('new',newimg)
	#cv2.imshow('mask',mask)
	#cv2.imshow('compare',shape)
	#cv2.imshow('res',colored)

	return False

def actionOnGreenLight ():
	return

def actionOntFound ():
	return

while(doProcessing):

	isGreenLight = getGreenLight(cap)

	if isGreenLight:
		actionOnGreenLight()
	else:
		actionOntFound()

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
	    break

cv2.destroyAllWindows()
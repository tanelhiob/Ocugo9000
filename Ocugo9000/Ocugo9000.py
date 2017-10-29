import cv2
import numpy as np
import sys
import time
import math
import platform
from buzzer import Buzzer

#ORANGE 22
#YELLOW 38
#GREEN 75
#BLUE 130
#VIOLET 160
#RED 179

#debug
# print(cv2.__version__)

isPi =  not platform.system() == 'Windows'

red = (0, 0, 255)
green = (0, 255, 0)

buzzer = Buzzer(18, 50, 1, 0.5)
coordinates = [(0,0)] * 10

def createTemplateShape ():

	original = cv2.imread('./sabloon.jpg')

	image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
	lower = np.array([0, 0, 5])
	upper = np.array([179, 255, 255])
	mask = cv2.inRange(image, lower, upper)
	image = cv2.bitwise_and(image, image, mask = mask)
	kernel = np.ones((5,5),np.uint8)
	image = cv2.erode(image,kernel,iterations = 1)
	image = cv2.dilate(image,kernel,iterations = 1)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	if isPi:
		contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	else:
		_, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	#debug
#	cv2.drawContours(original,contours[0],-1,(0,0,255),1)
#	cv2.imshow('contours', original)

	if not len(contours) == 1:
		raise("template doesn't have exactly 1 contours, actual result: " + str(len(contours)))

	return contours[0];

def processFrameForCamera(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower = np.array([50,15,25])
	upper = np.array([65,255,255])
	mask = cv2.inRange(frame, lower, upper)
	frame = cv2.bitwise_and(frame, frame, mask = mask)
	kernel = np.ones((1, 1), np.uint8)
	frame = cv2.erode(frame, kernel, iterations = 1)
	frame = cv2.dilate(frame, kernel, iterations = 1)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	return frame

def processFrameForTrainingImages(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower = np.array([70,150,50])
	upper = np.array([95,255,255])
	mask = cv2.inRange(frame, lower, upper)
	frame = cv2.bitwise_and(frame, frame, mask = mask)
	frame = cv2.medianBlur(frame, 1)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	return frame

def getGreenLight (original, templateShape, processingFunction):

	frame = processingFunction(original)

	if isPi:
		contours, _ = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	else:
		_, contours, _ = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	probableContours = list(filter(lambda x: x.size > 25, contours))
	matches = list(map(lambda x: (x, cv2.matchShapes(x, templateShape, 1, 0.0)), probableContours))
	matches.sort(key = lambda x: x[1])
	
	#debug
	if len(matches) > 0 and matches[0][1] < 0.7:
		bestContours = list(map(lambda x: x[0], matches[:1]))
	
		M = cv2.moments(bestContours[0])
		cX = int(M['m10'] / M['m00'])
		cY = int(M['m01'] / M['m00'])
		currXY = (cX,cY)

		for i in range(len(coordinates)-1):
			coordinates[i+1] = coordinates[i]
		coordinates[0] = currXY
		
		avgX = sum(map(lambda (x,y): x, coordinates))*1.0 / len(coordinates)
		avgY = sum(map(lambda (x,y): y, coordinates))*1.0 / len(coordinates)
		distance = math.hypot(currXY[0] - avgX, currXY[1] - avgY)
		
		if distance < 25:
			cv2.drawContours(original, bestContours,-1,(0,0,255),1)
			
#	cv2.imshow('coloredOriginal', original)

	if len(matches) > 0:
		if matches[0][1] < 0.7 and distance < 25:
			return True
		else:
#			cv2.imshow('nomatch',original)
			return False
	else:
#		cv2.imshow('nomatch',original)
		return False


def actionOnGreenLight ():
#	print('#####')
	buzzer.start()
	return


def actionOnNotFound ():
#	print('-----')
	buzzer.stop()
	return


cap = cv2.VideoCapture(0)
templateShape = createTemplateShape()

smoothing = [0] * 10
while(True):
	_, frame = cap.read()
	isGreenLight = getGreenLight(frame, templateShape, processFrameForCamera)

	#isGreenLight = getGreenLight(cv2.imread('TrainingImages\\foor-1.jpeg'), templateShape, processFrameForTrainingImages)
	for i in range(len(smoothing)-1):
		smoothing[i+1] = smoothing[i]
	smoothing[0] = isGreenLight
		
	isSmooth = round((float(sum(smoothing)) / len(smoothing)),0) == 1  
	
	if isSmooth:
		actionOnGreenLight()
		time.sleep(0.1)
	else:
		actionOnNotFound()

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		buzzer.stop()
		buzzer.cleanup()
	    	break

cv2.destroyAllWindows()

import cv2
import numpy as np
import sys
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

buzzer = Buzzer(18, 432, 0.75, 0.5)

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
	#cv2.drawContours(original,contours[0],-1,(0,0,255),1)
	#cv2.imshow('contours', original)

	if not len(contours) == 1:
		raise("template doesn't have exactly 1 contours, actual result: " + str(len(contours)))

	return contours[0];

def processFrameForCamera(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower = np.array([45,20,40])
	upper = np.array([75,255,255])
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
	
	probableContours = list(filter(lambda x: x.size > 50, contours))
	matches = list(map(lambda x: (x, cv2.matchShapes(x, templateShape, 1, 0.0)), probableContours))
	matches.sort(key = lambda x: x[1])
	
	#debug
	if len(matches) > 0 and matches[0][1] < 0.5:
		print(matches[0][1])
		bestContours = list(map(lambda x: x[0], matches[:1]))
		cv2.drawContours(original, bestContours,-1,(0,0,255),1)
	cv2.imshow('coloredOriginal', original)

	if len(matches) > 0:
		if matches[0][1] < 0.5:
			return True
		else:
			return False
	else:
		return False


def actionOnGreenLight ():
	buzzer.start()
	return


def actionOnNotFound ():
	buzzer.stop()
	return


cap = cv2.VideoCapture(0)
templateShape = createTemplateShape()

while(True):

	_, frame = cap.read()
	isGreenLight = getGreenLight(frame, templateShape, processFrameForCamera)

	#isGreenLight = getGreenLight(cv2.imread('TrainingImages\\foor-1.jpeg'), templateShape, processFrameForTrainingImages)

	if isGreenLight:
		actionOnGreenLight()
	else:
		actionOnNotFound()

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
	    break

cv2.destroyAllWindows()

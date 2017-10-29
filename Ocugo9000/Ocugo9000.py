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
#coordinates = [(0,0)] * 10

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
		contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	else:
		_, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	#debug
#	cv2.drawContours(original,contours[0],-1,(0,0,255),1)
#	cv2.imshow('contours', original)

	if not len(contours) == 1:
		raise("template doesn't have exactly 1 contours, actual result: " + str(len(contours)))

	return contours[0];

def processFrameForCamera(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower = np.array([45,45,20])
	upper = np.array([55,255,255])
	mask = cv2.inRange(frame, lower, upper)
	frame = cv2.bitwise_and(frame, frame, mask = mask)
	kernel = np.ones((3, 3), np.uint8)
	#frame = cv2.erode(frame, kernel, iterations = 1)
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
#	cv2.imshow('frame',frame)
	if isPi:
		contours, _ = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	else:
		_, contours, _ = cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	probableContours = list(filter(lambda x: x.nbytes > 100, contours))
	matches = list(map(lambda x: (x, cv2.matchShapes(x, templateShape, cv2.cv.CV_CONTOURS_MATCH_I3, 1)), probableContours))
	matches.sort(key = lambda x: x[1])
	
	#debug
	if len(matches) > 0:# and matches[0][1] < 1:
		bestContours = list(map(lambda x: x[0], matches[:1]))
#		print(matches[0][1])
#		M = cv2.moments(bestContours[0])
#		cX = int(M['m10'] / M['m00'])
#		cY = int(M['m01'] / M['m00'])
#		currXY = (cX,cY)

#		for i in range(len(coordinates)-1):
#			coordinates[i+1] = coordinates[i]
#		coordinates[0] = currXY
		
#		avgX = sum(map(lambda (x,y): x, coordinates))*1.0 / len(coordinates)
#		avgY = sum(map(lambda (x,y): y, coordinates))*1.0 / len(coordinates)
#		distance = math.hypot(currXY[0] - avgX, currXY[1] - avgY)
		
#		if distance < 25:
#		cv2.drawContours(original, bestContours,-1,(0,0,255),1)
			
#	cv2.imshow('coloredOriginal', original)

	if len(matches) > 0:
		if matches[0][1] < 0.41: # and distance < 25:
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
cap.set(12, 0.5)
#cap.set(13, 25)
templateShape = createTemplateShape()

smoothing = [0] * 20
#t_dur = time.time() + 5
#counter = 0
while(True):# and time.time() < t_dur):
#	counter = counter + 1
	_, frame = cap.read()
	isGreenLight = getGreenLight(frame, templateShape, processFrameForCamera)

	#isGreenLight = getGreenLight(cv2.imread('TrainingImages\\foor-1.jpeg'), templateShape, processFrameForTrainingImages)
	for i in range(len(smoothing)-1):
		smoothing[i+1] = smoothing[i]
	smoothing[0] = isGreenLight
		
	average = round((float(sum(smoothing)) / len(smoothing)),0)  
#	print(average)
	isSmooth = average == 1
	if isSmooth:
		actionOnGreenLight()
#		time.sleep(0.2)
	else:
		actionOnNotFound()

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		buzzer.stop()
		buzzer.cleanup()
	    	break

#print('counter = ' + str(counter))
cap.set(cv2.cv.CV_CAP_PROP_SETTINGS,0.0);
cv2.VideoCapture(0).release()
cv2.destroyAllWindows()

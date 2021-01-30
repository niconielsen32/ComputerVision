import sys
import cv2 as cv
import numpy as np
import time


def add_HSV_filter(frame):

	# Blurring the frame
    blur = cv.GaussianBlur(frame,(5,5),0) 

    # Converting RGB to HSV
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    l_b = np.array([143, 112, 53])        # Lower limit for red ball
    u_b = np.array([255, 255, 255])       # Upper limit for red ball

	#l_b = np.array([140, 106, 0])        # LOWER LIMIT FOR BLUE COLOR!!!
	#u_b = np.array([255, 255, 255])

	# HSV-filter mask
    mask = cv.inRange(hsv, l_b, u_b)

    # Morphological Operation - Opening - Erode followed by Dilate - Remove noise
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    return mask

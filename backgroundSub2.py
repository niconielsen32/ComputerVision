import cv2 as cv
import numpy as np
import sys

def resize(dst,img):
	width = img.shape[1]
	height = img.shape[0]
	dim = (width, height)
	resized = cv.resize(dst, dim, interpolation = cv.INTER_AREA)
	return resized


video = cv.VideoCapture(0, cv.CAP_DSHOW)
oceanVideo = cv.VideoCapture("ocean.mp4")

takeBgImage = 0

ret, bgReference = video.read()

while True:
        ret, img = video.read()
        ret2, bg = oceanVideo.read()

        if bg is not None:
                bg = resize(bg,bgReference)

        if takeBgImage == 0:
                bgReference = img

        # Create a mask
        diff1=cv.subtract(img, bgReference)
        diff2=cv.subtract(bgReference, img)

        diff = diff1 + diff2
        diff[abs(diff) < 25.0] = 0

        cv.imshow("diff1", diff)

        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        gray[np.abs(gray) < 10] = 0
        fgMask = gray

        # Opening to reduce noise
        kernel = np.ones((3,3), np.uint8)

        fgMask = cv.erode(fgMask, kernel, iterations=2)
        fgMask = cv.dilate(fgMask, kernel, iterations=2)

        fgMask[fgMask>5]=255

        cv.imshow("Foreground Mask", fgMask)

        # Inverting the mask
        fgMask_inv = cv.bitwise_not(fgMask)

        # Get relevant information from fore- and background with the masks
        fgImage = cv.bitwise_and(img, img, mask=fgMask)
        bgImage = cv.bitwise_and(bg, bg, mask=fgMask_inv)

        # Combine the fore- and background images
        bgSub = cv.add(bgImage,fgImage)

        cv.imshow('Background Removed', bgSub)
        cv.imshow('Original', img)

        key = cv.waitKey(5) & 0xFF
        if ord('q') == key:
                break
        elif ord('e') == key:
                takeBgImage = 1
                print("Background Captured")
        elif ord('r') == key:
                takeBgImage = 0
                print("Ready to Capture new Background")


cv.destroyAllWindows()
video.release()
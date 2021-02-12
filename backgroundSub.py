import cv2 as cv
import numpy as np


#backSub = cv.createBackgroundSubtractorKNN()
backSub = cv.createBackgroundSubtractorMOG2()

capture = cv.VideoCapture(0, cv.CAP_DSHOW)

if not capture.isOpened():
    print('Unable to open')
    exit(0)


while True:
    ret, frame = capture.read()
    if frame is None:
        break

    frame = cv.resize(frame, (800,800))

    
    fgMask = backSub.apply(frame)


    kernel = np.ones((5,5), np.uint8)

    fgMask = cv.erode(fgMask, kernel, iterations=2)
    fgMask = cv.dilate(fgMask, kernel, iterations=2)
    
    
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    

    fgMask[np.abs(fgMask) < 250] = 0


    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
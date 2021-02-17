import cv2
import numpy as np
import time

# Background subtraction algortihm
backSub = cv2.createBackgroundSubtractorKNN()



# Start camera
video = cv2.VideoCapture(0);


fps = video.get(cv2.CAP_PROP_FPS)
print("Frames per second camera: {0}".format(fps))

# Number of frames to capture
num_frames = 1;

print("Capturing {0} frames".format(num_frames))


# Grab a few frames
while True:

    # Start time
    start = time.time()

    ret, frame = video.read()

    if frame is None:
        print("No frame")
        break

    fgMask = backSub.apply(frame)

    sum = 0
    N = 300
    for i in range(0, N):
        for j in range(0, N):
            sum += 1

    kernel = np.ones((5,5), np.uint8)

    fgMask = cv2.erode(fgMask, kernel, iterations=1)
    fgMask = cv2.dilate(fgMask, kernel, iterations=1)


    # End time for whole program running 120 frames
    end = time.time()

    # Time elapsed
    seconds = end - start
    #print ("Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    fps  = num_frames / seconds
    #print("Estimated frames per second : {0}".format(fps))

    cv2.putText(fgMask, "FPS: " + str(round(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255))

    cv2.imshow('fgMask', fgMask)



    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break



# Release camera
video.release()

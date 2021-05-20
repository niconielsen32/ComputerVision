import cv2
import mediapipe as mp
import time

faceDetector = mp.solutions.face_detection
drawing = mp.solutions.drawing_utils


# For webcam input:
cap = cv2.VideoCapture(0)

with faceDetector.FaceDetection(

    min_detection_confidence=0.5) as face_detection:

  while cap.isOpened():

    success, image = cap.read()

    start = time.time()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    if results.detections:
      for id, detection in enumerate(results.detections):
        drawing.draw_detection(image, detection)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    print("FPS: ", fps)

    cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

    cv2.imshow('MediaPipe Face Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
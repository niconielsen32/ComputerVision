import cv2
import time
import numpy as np


 
# Load the COCO class names
with open('models/Object/object_detection_classes_coco.txt', 'r') as f:
   class_names = f.read().split('\n')

print(class_names)
 
# Get a different colors for each of the classes
colors = np.random.uniform(0, 255, size=(len(class_names), 3))
 
# Load the DNN model
model = cv2.dnn.readNet(model='models/Object/frozen_inference_graph.pb', config='models/Object/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', framework='TensorFlow')

# Set backend and target to CUDA to use GPU
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
 
# Webcam
cap = cv2.VideoCapture(0)


min_confidence_score = 0.6


while cap.isOpened():

    succes, img = cap.read()
   
    imgHeight, imgWidth, channels = img.shape

    # create blob from image
    blob = cv2.dnn.blobFromImage(img, size=(300, 300), mean=(104, 117, 123), swapRB=True)

    # start time to calculate FPS
    start = time.time()

    # Set input to the model
    model.setInput(blob)

    # Make forward pass in model
    output = model.forward()   

    # End time
    end = time.time()

    # calculate the FPS for current frame detection
    fps = 1 / (end-start)


    # loop over each of the detections
    for detection in output[0, 0, :, :]:

        # extract the confidence of the detection
        confidence = detection[2]

        # draw bounding boxes only if the detection confidence is above...
        # ... a certain threshold, else skip
        if confidence > min_confidence_score:
            # get the class id
            class_id = detection[1]

            # map the class id to the class
            class_name = class_names[int(class_id)-1]
            color = colors[int(class_id)]

            # Bounding box coordinates
            bboxX = detection[3] * imgWidth
            bboxY = detection[4] * imgHeight
            # Bounding box width and height
            bboxWidth = detection[5] * imgWidth
            bboxHeight = detection[6] * imgHeight

            # Draw a rectangle around each detected object
            cv2.rectangle(img, (int(bboxX), int(bboxY)), (int(bboxWidth), int(bboxHeight)), color, thickness=2)
            # Put the class name text on the detected object
            cv2.putText(img, class_name, (int(bboxX), int(bboxY - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    
    # Show FPS
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('image', img)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

 

cap.release()
cv2.destroyAllWindows()
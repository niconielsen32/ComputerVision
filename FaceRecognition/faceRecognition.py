import cv2
import face_recognition
import numpy as np



image = face_recognition.load_image_file("C:/Users/nhoei/barackObama.jpg")

cv2.imshow("Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)


# Face Detection
face_locations = face_recognition.face_locations(image)

for (top, right, bottom, left) in face_locations:
    # Draw a box around the face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)


cv2.imshow("Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)



# Facial Landmarks Detection
face_landmarks_list = face_recognition.face_landmarks(image)

facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip']


for face_landmarks in face_landmarks_list:
    for facial_feature in facial_features:
        for point in face_landmarks[facial_feature]:
            image = cv2.circle(image, point, 2, (255,60,170),2)
            
            
cv2.imshow("Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)


# Face Recognition
image = face_recognition.load_image_file("C:/Users/nhoei/barackObama.jpg")
#unknown_image = face_recognition.load_image_file("C:/Users/nhoei/barackObama2.jpg")
#unknown_image = face_recognition.load_image_file("C:/Users/nhoei/elonMusk.jpg")
unknown_image = face_recognition.load_image_file("C:/Users/nhoei/tigerWoods.jpeg")

image_encoding = face_recognition.face_encodings(image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([image_encoding], unknown_encoding)

cv2.putText(unknown_image, f'Barack Obama: {results[0]}', (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.imshow("Barack Obama", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imshow("Unknown", cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)

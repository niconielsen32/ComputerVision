import numpy as np
import cv2
import easyocr

reader = easyocr.Reader(['en'])
ocr_results = reader.readtext("C:/Users/nhoei/image.png")

print(ocr_results)

top_left = ocr_results[0][0][0]
bottom_right = ocr_results[0][0][2]

text = ocr_results[0][1]

img = cv2.imread("C:/Users/nhoei/image.png")
img = cv2.rectangle(img, top_left, bottom_right, (0,0,255), 5)
img = cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)


cv2.imshow("img", img)
cv2.waitKey(0)


img = cv2.imread("C:/Users/nhoei/licenseplate.jpg")
ocr_results = reader.readtext("C:/Users/nhoei/licenseplate.jpg")
print(ocr_results)

confidence_treshold = 0.2

for detection in ocr_results:
    if detection[2] > confidence_treshold:
        top_left = [int(value) for value in detection[0][0]]
        bottom_right = [int(value) for value in detection[0][2]]
        text = detection[1]
        img = cv2.rectangle(img, top_left, bottom_right, (0,0,255), 5)
        img = cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)


cv2.imshow("img", img)
cv2.waitKey(0)
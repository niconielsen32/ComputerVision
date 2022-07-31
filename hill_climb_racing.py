import numpy as np
import mediapipe as mp
import cv2
import time

import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4) as hands:

    while cap.isOpened():

        success, image = cap.read()
        
        h, w, c = image.shape

        start = time.perf_counter()
   

        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        # Process the image and find hands
        results = hands.process(image)

        image.flags.writeable = True

        # Draw the hand annotations on the image.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
            
            
            index_finger_tip = hand_landmarks.landmark[0]
            
            index_finger_tip_x = index_finger_tip.x * w
            index_finger_tip_y = index_finger_tip.y * h
            
            if index_finger_tip_x > w/2:
                cv2.putText(image, "Gas", (500,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                pyautogui.keyDown('right')
                pyautogui.keyUp('left')
            elif index_finger_tip_x < w/2:
                cv2.putText(image, "Brake", (500,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
                pyautogui.keyDown('left')
                pyautogui.keyUp('right')
            

        cv2.line(image, (int(w/2), 0), (int(w/2), h), (0, 255, 0), 2)

        end = time.perf_counter()
        totalTime = end - start

        fps = 1 / totalTime

        cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        cv2.imshow('MediaPipe Hands', image)



        if cv2.waitKey(5) & 0xFF == 27:
          break

cap.release()

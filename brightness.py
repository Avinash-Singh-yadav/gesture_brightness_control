import screen_brightness_control as sbc
import mediapipe as mp
import cv2
import numpy as np
from math import hypot

# Initialize webcam
cam_clip = cv2.VideoCapture(0)
if not cam_clip.isOpened():
    print("Error: Could not open webcam.")
    exit()

my_Hands = mp.solutions.hands
hands = my_Hands.Hands()
Hand_straight_line_draw = mp.solutions.drawing_utils

while True:
    success, img = cam_clip.read()
    if not success:
        print("Failed to grab frame")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            Hand_straight_line_draw.draw_landmarks(img, handlandmark, my_Hands.HAND_CONNECTIONS)

    if lmList:
        # Thumb tip = 4, Index finger tip = 8
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 8, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = hypot(x2 - x1, y2 - y1)

        # Map length to brightness (15-220 maps to 0-100)
        bright = np.interp(length, [15, 220], [0, 100])
        print(f"Brightness: {bright:.2f}, Length: {length:.2f}")

        try:
            sbc.set_brightness(int(bright))
        except Exception as e:
            print(f"Error setting brightness: {e}")

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam_clip.release()
cv2.destroyAllWindows()

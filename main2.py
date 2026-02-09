import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Screen size
screen_width, screen_height = pyautogui.size()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []

            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Finger tip y-positions
            index_y = lm_list[8][2]
            middle_y = lm_list[12][2]
            ring_y = lm_list[16][2]
            pinky_y = lm_list[20][2]

            # Palm reference
            palm_y = lm_list[0][2]

            # Open palm → scroll up
            if index_y < palm_y and middle_y < palm_y and ring_y < palm_y and pinky_y < palm_y:
                pyautogui.scroll(40)
                cv2.putText(frame, "Scrolling Up",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            # Closed fist → scroll down
            if index_y > palm_y and middle_y > palm_y and ring_y > palm_y and pinky_y > palm_y:
                pyautogui.scroll(-40)
                cv2.putText(frame, "Scrolling Down",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Gesture Controlled Scrolling", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
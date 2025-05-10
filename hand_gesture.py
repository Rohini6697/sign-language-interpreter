import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def get_label(landmarks):
    threshold = 0.04

    def is_finger_up(tip_id, pip_id):
        return landmarks.landmark[tip_id].y < landmarks.landmark[pip_id].y - threshold

    
    thumb_up = landmarks.landmark[4].x < landmarks.landmark[3].x

    index_up = is_finger_up(8, 6)
    middle_up = is_finger_up(12, 10)
    ring_up = is_finger_up(16, 14)
    pinky_up = is_finger_up(20, 18)

    
    if thumb_up and not (index_up or middle_up or ring_up or pinky_up):
        return "Thumbs Up"

    
    if index_up and not (thumb_up or middle_up or ring_up or pinky_up):
        return "Number 1"

    return "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 640))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            label = get_label(landmarks)
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

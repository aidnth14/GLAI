import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
from collections import deque
import pyttsx3

# ===== Load Model =====
model = joblib.load("asl_model.pkl")

# ===== Text-to-Speech Setup =====
engine = pyttsx3.init()

# ===== Setup MediaPipe =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ===== Camera Setup =====
cap = cv2.VideoCapture(0)

# ===== Prediction Controls =====
last_prediction_time = time.time()
DELAY_SECONDS = 1.5
prediction_history = deque(maxlen=5)
final_prediction = "..."
sentence = ""

# ===== Main Loop =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Extract 63 landmarks
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                current_time = time.time()
                if current_time - last_prediction_time > DELAY_SECONDS:
                    prediction = model.predict([landmarks])[0]
                    prediction_history.append(prediction)
                    final_prediction = max(set(prediction_history), key=prediction_history.count)
                    last_prediction_time = current_time
    else:
        final_prediction = "No hand"
        prediction_history.clear()

    # ===== Key Controls =====
    if key == ord('s'):  # Add predicted word to sentence
        if final_prediction not in ["...", "No hand"]:
            sentence += final_prediction + " "

    elif key == ord('b'):  # Backspace last word
        words = sentence.strip().split(" ")
        sentence = " ".join(words[:-1]) + " " if len(words) > 1 else ""

    elif key == 13:  # Enter key to speak sentence
        engine.say(sentence.strip())
        engine.runAndWait()

    elif key == ord('q'):  # Quit
        break

    # ===== Display Info =====
    cv2.putText(frame, f"Word: {final_prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("ASL Translator - Sentence Builder", frame)

cap.release()
cv2.destroyAllWindows()

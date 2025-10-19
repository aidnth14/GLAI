import cv2
import mediapipe as mp
import csv
import os

LABEL = "Hello"  # 👈 Change this to B, C, Hello, etc.
CSV_FILE = f"{LABEL}_data.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with open(CSV_FILE, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # Get landmarks
                landmark_list = []
                for lm in handLms.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])

                # Show status
                cv2.putText(frame, f"Press 's' to save [{LABEL}]", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Collecting Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and results.multi_hand_landmarks:
            csv_writer.writerow(landmark_list)
            print(f"[✓] Saved one sample for '{LABEL}'")

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

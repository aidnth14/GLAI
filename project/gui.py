import tkinter as tk
from tkinter import ttk, END
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
from collections import deque
import pyttsx3
import threading

# --- Main GUI Application Class ---
class ASL_GUI_App:
    def __init__(self, window, window_title="Live ASL Prediction GUI"):
        self.window = window
        self.window.title(window_title)
        
        # === 1. Load Dependencies ===
        # MAKE SURE 'asl_model.pkl' IS IN THE SAME DIRECTORY
        self.model = joblib.load("asl_model.pkl") 
        self.engine = pyttsx3.init()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam. Please check your camera device.")

        # === 2. Prediction Controls ===
        self.last_prediction_time = time.time()
        self.DELAY_SECONDS = 1.5
        self.prediction_history = deque(maxlen=5)
        self.final_prediction = "..."
        self.sentence = ""
        
        # Variable to hold the latest frame
        self.current_frame = None
        self.stop_event = threading.Event()
        
        # === 3. GUI Layout (Tkinter) ===
        
        # A. Video Panel
        self.video_frame = ttk.Frame(window, padding="10")
        self.video_frame.pack(side="left", padx=10, pady=10)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()
        
        # B. Controls and Output Panel
        self.controls_frame = ttk.Frame(window, padding="10")
        self.controls_frame.pack(side="right", fill="y", padx=10, pady=10)
        
        # Current Word Prediction
        ttk.Label(self.controls_frame, text="Current Word:", font=('Arial', 14, 'bold')).pack(pady=(0, 5), anchor='w')
        self.word_label = ttk.Label(self.controls_frame, text=self.final_prediction, font=('Arial', 18), foreground='green')
        self.word_label.pack(pady=(0, 20), anchor='w')

        # Sentence Display
        ttk.Label(self.controls_frame, text="Sentence:", font=('Arial', 14, 'bold')).pack(pady=(10, 5), anchor='w')
        self.sentence_display = tk.Entry(self.controls_frame, width=40, font=('Arial', 12))
        self.sentence_display.pack(pady=(0, 20), ipady=5)

        # C. Action Buttons (now also show key bindings)
        ttk.Button(self.controls_frame, text="Add Word (S)", command=self.add_word_action, width=20).pack(pady=5)
        ttk.Button(self.controls_frame, text="Backspace Word (B)", command=self.backspace_word_action, width=20).pack(pady=5)
        ttk.Button(self.controls_frame, text="Speak Sentence (Enter)", command=self.speak_sentence_action, width=20).pack(pady=5)
        ttk.Button(self.controls_frame, text="Quit (Q)", command=self.on_closing, width=20, style='TButton').pack(pady=30)
        
        # === 4. Key Bindings (The Fix) ===
        # Binds keyboard keys to the methods. The methods must accept an event argument.
        self.window.bind('s', self.add_word_action)
        self.window.bind('b', self.backspace_word_action)
        self.window.bind('<Return>', self.speak_sentence_action) # <Return> is the Enter key
        self.window.bind('q', self.on_closing)
        
        # === 5. Start Threads and mainloop ===
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.after(10, self.update_gui)
        self.window.mainloop()

    # --- Video Loop (runs in a separate thread) ---
    def video_loop(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            self.current_frame = self.process_frame(frame)
            time.sleep(0.01)

    # --- Frame Processing (The core logic from predict_live.py) ---
    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)
                
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                current_time = time.time()
                if current_time - self.last_prediction_time > self.DELAY_SECONDS:
                    prediction_vector = self.model.predict([landmarks])[0]
                    prediction = prediction_vector # Replace this with your actual label extraction/mapping
                    
                    self.prediction_history.append(prediction)
                    self.final_prediction = max(set(self.prediction_history), key=self.prediction_history.count)
                    self.last_prediction_time = current_time
        else:
            self.final_prediction = "No hand"
            self.prediction_history.clear()

        cv2.putText(frame, f"Word: {self.final_prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv2_image)

    # --- GUI Update (runs in the main thread) ---
    def update_gui(self):
        if self.current_frame:
            img = self.current_frame.resize((640, 480))
            self.photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=self.photo)
            self.video_label.image = self.photo 
            
        self.word_label.config(text=self.final_prediction)

        if not self.stop_event.is_set():
            self.window.after(10, self.update_gui)

    # --- Button Command Actions (Updated to accept event=None) ---
    def add_word_action(self, event=None):
        """Simulates the 's' key: Add predicted word to sentence."""
        if self.final_prediction not in ["...", "No hand"]:
            self.sentence += self.final_prediction + " "
            self.sentence_display.delete(0, END)
            self.sentence_display.insert(0, self.sentence.strip())

    def backspace_word_action(self, event=None):
        """Simulates the 'b' key: Backspace last word."""
        words = self.sentence.strip().split(" ")
        # Adjusted logic: If there are words, remove the last one. If only one word, clear the sentence.
        if len(words) > 1 and words[-1]: 
            self.sentence = " ".join(words[:-1]) + " "
        elif len(words) == 1 and words[0]:
            self.sentence = ""
        else:
            self.sentence = ""
            
        self.sentence_display.delete(0, END)
        self.sentence_display.insert(0, self.sentence.strip())

    def speak_sentence_action(self, event=None):
        """Simulates the 'Enter' key: Speak sentence."""
        text_to_speak = self.sentence_display.get().strip() # Use the text in the entry box
        if text_to_speak:
            self.engine.say(text_to_speak)
            # Run in a separate thread to prevent the GUI from freezing
            threading.Thread(target=self.engine.runAndWait, daemon=True).start()

    def on_closing(self, event=None):
        """Handles closing the app and cleaning up resources."""
        self.stop_event.set()
        # It's important to allow the video thread time to stop gracefully
        if self.thread.is_alive():
            self.thread.join(timeout=1) 
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    # Remember to install the required libraries: 
    # pip install opencv-python mediapipe joblib pyttsx3 pillow
    root = tk.Tk()
    app = ASL_GUI_App(root)
    
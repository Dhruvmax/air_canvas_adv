import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import os
import pyaudio
from vosk import Model, KaldiRecognizer

# Initialize MediaPipe with more robust settings
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Canvas and color settings
canvas = np.zeros((480, 640, 3), dtype=np.uint8) + 255
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Red, Green, Blue, Yellow
color_names = ["RED", "GREEN", "BLUE", "YELLOW"]
color_index = 0

# Navbar settings (fit within 640 width)
navbar_height = 70
button_y = 5
navbar_buttons = {
    "RED": (10, button_y, 60, 60),
    "GREEN": (80, button_y, 60, 60),
    "BLUE": (150, button_y, 60, 60),
    "YELLOW": (220, button_y, 60, 60),
    "CLEAR": (290, button_y, 60, 60),
    "EYE": (360, button_y, 60, 60),
    "HAND": (430, button_y, 60, 60),
    "PEN": (500, button_y, 60, 60),
    "REC": (570, button_y, 60, 60)
}

# State variables
mode = 'eye_tracking'  # Options: 'eye_tracking', 'hand_gesture', 'pen_mode'
recording = False
caption_text = ""
caption_timeout = 0
stop_recognition = threading.Event()
recognition_thread = None
prev_x, prev_y = None, None
hover_time = 0
hover_target = None
mouse_x, mouse_y = -1, -1

# Initialize Vosk model
MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Vosk model not found at '{MODEL_PATH}'. Download from https://alphacephei.com/vosk/models")
    exit(1)
vosk_model = Model(MODEL_PATH)

# Initialize PyAudio
p = pyaudio.PyAudio()
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096
stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

def speech_recognition_thread():
    global caption_text, caption_timeout
    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    stream.start_stream()
    
    while not stop_recognition.is_set():
        try:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = rec.Result()
                text = eval(result).get("text", "")
                print(f"Final: {text}")
                if text.strip():
                    caption_text = text.capitalize()
                    caption_timeout = time.time() + 3
            else:
                partial = rec.PartialResult()
                partial_text = eval(partial).get("partial", "")
                if partial_text.strip():
                    caption_text = partial_text.capitalize()
                    caption_timeout = time.time() + 1
        except Exception as e:
            caption_text = f"Speech error: {str(e)[:20]}"
            caption_timeout = time.time() + 3
            time.sleep(0.5)
    
    stream.stop_stream()

def toggle_speech_recognition():
    global recording, recognition_thread, stop_recognition
    if recording:
        stop_recognition.set()
        if recognition_thread and recognition_thread.is_alive():
            recognition_thread.join(timeout=1)
        recording = False
    else:
        stop_recognition.clear()
        recording = True
        recognition_thread = threading.Thread(target=speech_recognition_thread)
        recognition_thread.daemon = True
        recognition_thread.start()

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y

def draw_navbar(frame, cursor_x, cursor_y):
    for i in range(navbar_height):
        alpha = i / navbar_height
        color = (int(50 + 50 * alpha), int(50 + 50 * alpha), int(50 + 50 * alpha))
        cv2.line(frame, (0, i), (frame.shape[1], i), color)

    for button_name, (bx, by, bw, bh) in navbar_buttons.items():
        if button_name == "EYE":
            base_color = (255, 0, 0) if mode == 'eye_tracking' else (100, 100, 100)
            text = "Eye"
        elif button_name == "HAND":
            base_color = (0, 255, 0) if mode == 'hand_gesture' else (100, 100, 100)
            text = "Hand"
        elif button_name == "PEN":
            base_color = (255, 165, 0) if mode == 'pen_mode' else (100, 100, 100)
            text = "Pen"
        elif button_name == "REC":
            base_color = (0, 0, 255) if not recording else (255, 0, 255)
            text = "Rec" if not recording else "Stop"
        elif button_name == "CLEAR":
            base_color = (100, 100, 100)
            text = "Clear"
        else:
            base_color = colors[color_names.index(button_name)]
            text = button_name[:3]
        
        is_hovered = bx <= cursor_x <= bx + bw and by <= cursor_y <= by + bh
        button_color = tuple(min(255, c + 50) for c in base_color) if is_hovered else base_color
        
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), button_color, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
        cv2.putText(frame, text, (bx + 5, by + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def draw_caption(frame):
    if caption_text and time.time() < caption_timeout:
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, frame.shape[0] - 70), (frame.shape[1] - 20, frame.shape[0] - 20), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        
        if recording and not caption_text.strip():
            cv2.putText(frame, "Listening...", (30, frame.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            words = caption_text.split()
            lines = [""]
            for word in words:
                test_line = lines[-1] + " " + word if lines[-1] else word
                if cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] < 600:
                    lines[-1] = test_line
                else:
                    lines.append(word)
            for i, line in enumerate(lines[:2]):
                cv2.putText(frame, line, (30, frame.shape[0] - 55 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Setup window with mouse callback
cv2.namedWindow('Webcam')
cv2.setMouseCallback('Webcam', mouse_callback)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        x, y = -1, -1
        should_draw = False  # Flag for pen mode drawing
        if mode == 'eye_tracking':
            try:
                results = face_mesh.process(frame_rgb)
                if results and results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        right_eye_outer = face_landmarks.landmark[133]
                        x = int(right_eye_outer.x * frame.shape[1])
                        y = int(right_eye_outer.y * frame.shape[0])
            except Exception as e:
                print(f"Face mesh processing error: {str(e)}")
                x, y = -1, -1
        elif mode == 'hand_gesture':
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x = int(index_tip.x * frame.shape[1])
                    y = int(index_tip.y * frame.shape[0])
        elif mode == 'pen_mode':
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    # Calculate distance between thumb and index finger tips
                    thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                    index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                    distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                    print(f"Distance between thumb and index: {distance}")  # Debug
                    # Use index finger tip as cursor, draw only if distance is small
                    x, y = index_x, index_y
                    should_draw = distance < 50  # Threshold for "close together"

        # Use mouse position if no tracked object, or combine with tracking
        cursor_x, cursor_y = x if x != -1 else mouse_x, y if y != -1 else mouse_y
        if cursor_x == -1 and cursor_y == -1:
            cursor_x, cursor_y = mouse_x, mouse_y

        draw_navbar(frame, cursor_x, cursor_y)

        # Handle hover actions
        if cursor_x != -1 and cursor_y != -1 and cursor_y <= navbar_height:
            for button_name, (bx, by, bw, bh) in navbar_buttons.items():
                if bx <= cursor_x <= bx + bw and by <= cursor_y <= by + bh:
                    if hover_target == button_name:
                        if time.time() - hover_time > 0.5:
                            if button_name == "EYE":
                                mode = 'eye_tracking'
                                print("Switched to Eye mode")
                            elif button_name == "HAND":
                                mode = 'hand_gesture'
                                print("Switched to Hand mode")
                            elif button_name == "PEN":
                                mode = 'pen_mode'
                                print("Switched to Pen mode")
                            elif button_name == "CLEAR":
                                canvas.fill(255)
                                print("Canvas cleared")
                            elif button_name == "REC":
                                toggle_speech_recognition()
                                print("Recording toggled")
                            elif button_name in color_names:
                                color_index = color_names.index(button_name)
                                print(f"Color changed to {button_name}")
                            hover_time = time.time()
                    else:
                        hover_target = button_name
                        hover_time = time.time()
                    break
            else:
                hover_target = None
        else:
            hover_target = None

        # Drawing logic
        if cursor_x != -1 and cursor_y != -1:
            cv2.circle(frame, (cursor_x, cursor_y), 8, (0, 255, 255), -1)
            if cursor_y > navbar_height and 0 <= cursor_x < 640 and 0 <= cursor_y - navbar_height < 480:
                canvas_y = cursor_y - navbar_height
                if mode == 'pen_mode' and should_draw:
                    cv2.circle(canvas, (cursor_x, canvas_y), 5, colors[color_index], -1)
                elif mode != 'pen_mode' and prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (cursor_x, canvas_y), colors[color_index], 5)
                prev_x, prev_y = cursor_x, canvas_y
            else:
                prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None

        draw_caption(frame)
        cv2.imshow('Webcam', frame)
        cv2.imshow('Canvas', canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if recording:
        stop_recognition.set()
        if recognition_thread and recognition_thread.is_alive():
            recognition_thread.join(timeout=1)
    stream.close()
    p.terminate()
    cap.release()
    cv2.destroyAllWindows()
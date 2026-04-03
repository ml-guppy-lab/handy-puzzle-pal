
import os
import urllib

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Hand connection pairs (MediaPipe standard)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # Thumb
    (0,5),(5,6),(6,7),(7,8),          # Index
    (0,9),(9,10),(10,11),(11,12),     # Middle
    (0,13),(13,14),(14,15),(15,16),   # Ring
    (0,17),(17,18),(18,19),(19,20),   # Pinky
    (5,9),(9,13),(13,17),             # Palm
]

def draw_hand_landmarks(frame, hand_landmarks):
    """Draw hand skeleton on frame using OpenCV."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 4, (0, 150, 200), 1)

# ============================================================================
# Initialize Video Capture
# ============================================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera. Check permissions.")
    exit(1)

ret, frame = cap.read()
if ret:
    frame_h, frame_w, _ = frame.shape
else:
    print("Failed to read frame")
    exit(1)

# ============================================================================
# Configure MediaPipe Hand Landmarker (Modern Tasks API)
# ============================================================================
# Download the model once: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
# Place it in the same folder as this script
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading hand detection model (one-time setup)...")
    MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)   # downloads the file and saves it locally
    print("Model downloaded successfully!")
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,                          # Important: Track BOTH hands
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

detector = vision.HandLandmarker.create_from_options(options)

# ============================================================================
# STEP 3: Helper Functions (Finger Detection)
# ============================================================================
def is_finger_up(landmarks, tip_idx, pip_idx):
    """True if finger tip is higher (smaller y) than PIP joint"""
    return landmarks[tip_idx].y < landmarks[pip_idx].y

def count_open_fingers(hand_landmarks):
    """Count how many fingers are open (0 to 5)"""
    if not hand_landmarks:
        return 0
    
    # Landmark indices: tip and pip for each finger
    fingers = [
        (8, 6),   # Index
        (12, 10), # Middle
        (16, 14), # Ring
        (20, 18), # Pinky
        (4, 3)    # Thumb (slightly different logic, but this works well)
    ]
    
    open_count = 0
    for tip, pip in fingers:
        if is_finger_up(hand_landmarks, tip, pip):
            open_count += 1
    return open_count

def are_both_hands_fully_open(detection_result):
    """Check if we have exactly 2 hands and both have all 5 fingers open"""
    if not detection_result.hand_landmarks or len(detection_result.hand_landmarks) != 2:
        return False
    
    for hand_lm in detection_result.hand_landmarks:
        if count_open_fingers(hand_lm) != 5:
            return False
    return True

# ============================================================================
# STEP 4: Main Loop
# ============================================================================
print("="*70)
print("Handy Puzzle Pal - Step 0")
print("Show BOTH hands with ALL fingers open → Magic starts!")
print("Press 'q' to quit")
print("="*70)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)  # Mirror for natural feel
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hands
    detection_result = detector.detect(mp_image)

    # Draw landmarks on both hands
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            draw_hand_landmarks(frame, hand_landmarks)

    # Check magic condition
    if are_both_hands_fully_open(detection_result):
        cv2.putText(frame, "BOTH HANDS OPEN! ✨", 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        # Later we will draw the rectangle here
    else:
        cv2.putText(frame, "Show both hands fully open", 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Handy Puzzle Pal - Step 0 | The ML Guppy", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================================
# Cleanup
# ============================================================================
detector.close()
cap.release()
cv2.destroyAllWindows()
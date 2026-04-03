
import os
import urllib
import random

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

def draw_hand_landmarks(frame, hand_landmarks, color=(0, 200, 255)):
    """Draw hand skeleton on frame using OpenCV."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 4, color, 1)

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
# STEP 2: Rectangle + Snapshot State
# ============================================================================
original_img = None          # Stores the cropped snapshot
snapshot_timer = 0           # cv2.getTickCount() timestamp of last snapshot
SNAPSHOT_DISPLAY_TICKS = int(cv2.getTickFrequency() * 2)  # 2 seconds in ticks
frozen_square = None         # (x1, y1, x2, y2) locked after snapshot
puzzle_img = None            # Shuffled 3x3 puzzle image

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

def get_bounding_rect(all_hand_landmarks, frame_w, frame_h):
    """Return (x1, y1, x2, y2) bounding box covering all landmarks of all hands."""
    xs = [int(lm.x * frame_w) for hand in all_hand_landmarks for lm in hand]
    ys = [int(lm.y * frame_h) for hand in all_hand_landmarks for lm in hand]
    return min(xs), min(ys), max(xs), max(ys)

def get_pinch(hand_landmarks, frame_w, frame_h, threshold=30):
    """Return (is_pinching, midpoint_px) for a single hand."""
    thumb = hand_landmarks[4]
    index = hand_landmarks[8]
    tx, ty = int(thumb.x * frame_w), int(thumb.y * frame_h)
    ix, iy = int(index.x * frame_w), int(index.y * frame_h)
    dist = np.hypot(tx - ix, ty - iy)
    mid = ((tx + ix) // 2, (ty + iy) // 2)
    return dist < threshold, mid

def make_puzzle(img):
    """Split img into a 3x3 grid, shuffle tiles, return assembled puzzle image."""
    # Resize to a multiple of 3 to ensure clean equal tiles
    size = 300  # 300x300 → each tile is 100x100
    img_sq = cv2.resize(img, (size, size))
    tile_size = size // 3

    # Slice into 9 tiles (row-major order)
    tiles = [
        img_sq[r * tile_size:(r + 1) * tile_size,
               c * tile_size:(c + 1) * tile_size]
        for r in range(3)
        for c in range(3)
    ]

    random.shuffle(tiles)

    # Assemble shuffled tiles back into a 3x3 grid
    rows = [
        np.hstack(tiles[r * 3:(r + 1) * 3])
        for r in range(3)
    ]
    puzzle = np.vstack(rows)
    return puzzle

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

    # ── STEP 2: Puzzle overlay (always rendered if snapshot exists) ──────────
    if frozen_square is not None:
        # Paste puzzle onto frame unconditionally — stays regardless of hand state
        x1, y1, x2, y2 = frozen_square
        side_w, side_h = x2 - x1, y2 - y1
        puzzle_resized = cv2.resize(puzzle_img, (side_w, side_h))
        frame[y1:y2, x1:x2] = puzzle_resized
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Draw landmarks on both hands (always on top of puzzle)
    if detection_result.hand_landmarks:
        landmark_color = (0, 0, 255) if frozen_square is not None else (0, 200, 255)
        h, w = frame.shape[:2]
        for hand_landmarks in detection_result.hand_landmarks:
            draw_hand_landmarks(frame, hand_landmarks, landmark_color)
            # ── STEP 4: Pinch detection ──────────────────────────────────────
            is_pinching, mid = get_pinch(hand_landmarks, w, h)
            if is_pinching:
                cv2.circle(frame, mid, 16, (0, 255, 255), -1)   # filled yellow dot
                cv2.circle(frame, mid, 18, (0, 180, 180), 2)    # teal outline
                cv2.putText(frame, "PINCH", (mid[0] + 20, mid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ── Hand gesture UI (only when no snapshot yet) ───────────────────────
    if frozen_square is None:
        if are_both_hands_fully_open(detection_result):
            cv2.putText(frame, "BOTH HANDS OPEN!",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

            h, w = frame.shape[:2]
            screen_area = w * h

            x1, y1, x2, y2 = get_bounding_rect(detection_result.hand_landmarks, w, h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            rect_w, rect_h = x2 - x1, y2 - y1
            side = max(rect_w, rect_h)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w - 1, x1 + side)
            y2 = min(h - 1, y1 + side)
            rect_area = (x2 - x1) * (y2 - y1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            pct = rect_area / screen_area * 100
            cv2.putText(frame, f"Area: {pct:.1f}%  (need >= 40%)",
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if rect_area >= 0.4 * screen_area:
                original_img = frame[y1:y2, x1:x2].copy()
                frozen_square = (x1, y1, x2, y2)
                snapshot_timer = cv2.getTickCount()
                puzzle_img = make_puzzle(original_img)
                print(f"Snapshot taken! Crop size: {original_img.shape[1]}x{original_img.shape[0]}")
        else:
            cv2.putText(frame, "Show both hands fully open",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show snapshot overlay for 2 seconds after capture
    if snapshot_timer and (cv2.getTickCount() - snapshot_timer) < SNAPSHOT_DISPLAY_TICKS:
        cv2.putText(frame, "SNAPSHOT TAKEN!",
                    (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 4)

    # Remind user how to reset once puzzle is showing
    if frozen_square is not None:
        cv2.putText(frame, "SPACE to reset",
                    (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (200, 200, 200), 2)

    # Show the frame
    cv2.imshow("Handy Puzzle Pal - Step 0 | The ML Guppy", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        # Reset everything — start fresh
        original_img = None
        frozen_square = None
        puzzle_img = None
        snapshot_timer = 0
        print("Reset! Show both hands again.")

# ============================================================================
# Cleanup
# ============================================================================
detector.close()
cap.release()
cv2.destroyAllWindows()
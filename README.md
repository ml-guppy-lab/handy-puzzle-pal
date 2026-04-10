# 🧩 Handy Puzzle Pal - The ML Guppy

**Turn your own face into a draggable 3x3 puzzle using nothing but your bare hands!**  
No mouse. No touchscreen. Just pure hand magic. ✨

![Demo Thumbnail](<img width="321" height="231" alt="Screenshot 2026-04-10 at 10 00 09 AM" src="https://github.com/user-attachments/assets/fd217880-426f-46e2-b06b-b7b943cadc1f" />)

## 🎥 Watch the Magic
[![Watch the full funny demo on Instagram](https://img.shields.io/badge/Watch_on_Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/reel/DW6izikv7FA/)

> "My face just became a puzzle in 0.5 seconds 😂" — The ML Guppy

## ✨ What It Does
1. Show both hands with all 10 fingers fully open
2. A glowing rectangle appears between your hands
3. When the rectangle gets big enough (~40% of screen) → **SNAP!** It captures your face
4. Your face instantly turns into a shuffled 3x3 puzzle
5. Pinch + drag pieces in the air to solve it like a real puzzle
6. Solve it → Fireworks and big "SOLVED!" celebration! 🎉

## 🛠️ Tech Stack
- **Python**
- **OpenCV** (for everything visual)
- **MediaPipe Hand Landmarker** (for super accurate hand + finger tracking)
- **NumPy** (for slicing your face into puzzle pieces)

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/ml-guppy-lab/handy-puzzle-pal.git
   cd handy-puzzle-pal/backend
   ```
2. Install dependencies
   pip install -r requirements.txt
3. Download the MediaPipe model:
  - Download hand_landmarker.task from here
  - Place it in the backend folder
4. Run the puzzle:
   python puzzle.py
5. Gesture Guide:
  - Both hands + all fingers open → Magic rectangle appears
  - Pinch (thumb + index) → Drag puzzle pieces
📸 Demo Video
Full chaotic demo → [Instagram Reel](https://img.shields.io/badge/Watch_on_Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/reel/DW6izikv7FA/)
<img width="321" height="231" alt="Screenshot 2026-04-10 at 10 00 09 AM" src="https://github.com/user-attachments/assets/8fbf5ced-abe6-4689-b00f-b6599993cdcf" />


👨‍💻 About The Creator
Sonal — **The ML Guppy**
From .NET engineer slowly diving deep into AI/ML by building fun & funny projects.

**Instagram**: [The ML Guppy](https://www.instagram.com/themlguppy/reels/)

**LinkedIn**: [Sonal Kumari](www.linkedin.com/in/sonalsh250)

**GitHub Org**: ml-guppy-lab

Made with love, chaos, and too much coffee ☕
If you build this, tag me on Instagram — I want to see your confused face getting puzzled! 😂
Star this repo if it made you smile! ⭐

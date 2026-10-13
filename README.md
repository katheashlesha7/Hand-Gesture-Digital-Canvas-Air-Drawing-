# Air Canvas

Draw on a digital canvas by moving your index finger in front of a webcam. Air Canvas uses Python, OpenCV, and MediaPipe's Hand Landmarker task to track a hand locally on your computer.

## Setup

Use Python 3.10–3.12. On Windows, install the Python packages from PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

On the first launch, Air Canvas downloads Google's `hand_landmarker.task` model from MediaPipe's public model storage and caches it in `.cache/`. The webcam feed and landmark processing stay on your computer.

## Draw in the air

- Show one hand in the live camera preview.
- Move your index finger to position the cursor on the board.
- Touch your thumb and index finger to draw. Release the pinch to lift the pen.
- Select a color or brush size from the tools panel. Erase mode uses the same gesture.
- Use **Undo stroke**, **Clear canvas**, or **Save artwork** to manage your drawing.
- Press **Space** to pause or resume drawing, **Ctrl+Z** to undo, and **Esc** to quit.

The preview is mirrored, and its index fingertip position maps directly to the canvas. If the camera does not start, check Windows camera permissions and close other apps using the webcam.

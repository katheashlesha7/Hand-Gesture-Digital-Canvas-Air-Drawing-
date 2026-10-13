"""Air Canvas: draw on a digital canvas with a webcam-tracked index finger."""

from __future__ import annotations

import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from tkinter import Canvas, DoubleVar, IntVar, StringVar, Tk, filedialog, messagebox, ttk
import tkinter as tk

try:
    from PIL import Image, ImageDraw, ImageTk
except ImportError as exc:  # pragma: no cover - surfaced in the setup guidance
    raise SystemExit("Pillow is missing. Run: python -m pip install -r requirements.txt") from exc


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / ".cache"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
BACKGROUND = "#F8FAFC"
INK = "#17212B"
MUTED = "#718096"
PANEL = "#FFFFFF"
LINE = "#E7EDF2"
ACCENT = "#12A887"
PALETTE = ["#17212B", "#12A887", "#4C7CF3", "#8B5CF6", "#F97358", "#F4A340"]


@dataclass
class TrackingSnapshot:
    frame: object | None = None
    point: tuple[float, float] | None = None
    drawing: bool = False
    hand_found: bool = False
    message: str = "Starting camera…"
    fps: float = 0.0


class CameraTracker:
    """Own the camera and MediaPipe task on a worker thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = TrackingSnapshot()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="air-canvas-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()

    def snapshot(self) -> TrackingSnapshot:
        with self._lock:
            return self._snapshot

    def _publish(self, **values: object) -> None:
        with self._lock:
            self._snapshot = TrackingSnapshot(**{**self._snapshot.__dict__, **values})

    @staticmethod
    def _ensure_model() -> Path:
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
            return MODEL_PATH
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        temporary_path = MODEL_PATH.with_suffix(".download")
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "AirCanvas/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response, temporary_path.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if temporary_path.stat().st_size < 100_000:
                raise RuntimeError("The hand tracking model download was incomplete.")
            temporary_path.replace(MODEL_PATH)
        finally:
            temporary_path.unlink(missing_ok=True)
        return MODEL_PATH

    def _run(self) -> None:
        capture = None
        landmarker = None
        try:
            import cv2
            import mediapipe as mp

            self._publish(message="Preparing hand tracking model…")
            model_path = self._ensure_model()
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

            capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
            if not capture.isOpened():
                raise RuntimeError("No webcam found. Check the camera connection and permissions.")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            previous_time = 0
            last_tick = time.perf_counter()
            fps = 0.0
            self._publish(message="Camera ready · show one hand")

            while self._running.is_set():
                ok, frame = capture.read()
                if not ok:
                    self._publish(message="Waiting for webcam frames…", hand_found=False, point=None, drawing=False)
                    time.sleep(0.04)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(previous_time + 1, int(time.monotonic() * 1000))
                previous_time = timestamp
                result = landmarker.detect_for_video(mp_image, timestamp)

                point = None
                drawing = False
                hand_found = bool(result.hand_landmarks)
                if hand_found:
                    landmarks = result.hand_landmarks[0]
                    height, width = frame.shape[:2]
                    pixels = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
                    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                    for connection in connections:
                        start = pixels[connection.start]
                        end = pixels[connection.end]
                        cv2.line(frame, start, end, (74, 206, 173), 2, cv2.LINE_AA)
                    for index, (x, y) in enumerate(pixels):
                        radius = 6 if index == 8 else 3
                        color = (64, 205, 167) if index == 8 else (239, 247, 246)
                        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
                    point = (float(landmarks[8].x), float(landmarks[8].y))
                    pinch_distance = ((landmarks[4].x - landmarks[8].x) ** 2 + (landmarks[4].y - landmarks[8].y) ** 2) ** 0.5
                    drawing = pinch_distance < 0.055
                    message = "Drawing · thumb and index pinched" if drawing else "Move index finger · pinch to draw"
                else:
                    message = "Show your hand to the camera"

                now = time.perf_counter()
                instant_fps = 1.0 / max(now - last_tick, 0.001)
                fps = instant_fps if fps == 0 else (fps * 0.84 + instant_fps * 0.16)
                last_tick = now
                self._publish(
                    frame=frame,
                    point=point,
                    drawing=drawing,
                    hand_found=hand_found,
                    message=message,
                    fps=fps,
                )
        except Exception as exc:
            self._publish(message=f"Camera unavailable · {exc}", hand_found=False, point=None, drawing=False)
        finally:
            if capture is not None:
                capture.release()
            if landmarker is not None:
                landmarker.close()


class AirCanvasApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Air Canvas · Draw in the air")
        self.root.geometry("1440x900")
        self.root.minsize(1120, 740)
        self.root.configure(bg=BACKGROUND)

        self.camera = CameraTracker()
        self.selected_color = StringVar(value=PALETTE[1])
        self.brush_size = IntVar(value=8)
        self.mode = StringVar(value="draw")
        self.status_text = StringVar(value="Starting camera…")
        self.fps_text = StringVar(value="-- FPS")
        self.stroke_records: list[dict[str, object]] = []
        self.active_stroke: dict[str, object] | None = None
        self.previous_canvas_point: tuple[float, float] | None = None
        self.filtered_point: tuple[float, float] | None = None
        self.last_snapshot: TrackingSnapshot | None = None
        self.photo = None
        self.preview_photo = None
        self._swatch_buttons: list[tuple[tk.Button, str]] = []

        self._configure_styles()
        self._build_layout()
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Control-Z>", lambda _event: self.undo())
        self.root.bind("<Control-Shift-S>", lambda _event: self.save_png())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.camera.start()
        self.root.after(30, self._update)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TScale", background=PANEL, troughcolor="#EAF0F3", sliderlength=16)

    def _build_layout(self) -> None:
        shell = tk.Frame(self.root, bg=BACKGROUND)
        shell.pack(fill="both", expand=True, padx=28, pady=(20, 24))
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_rowconfigure(1, weight=1)

        header = tk.Frame(shell, bg=BACKGROUND)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        brand = tk.Frame(header, bg=BACKGROUND)
        brand.pack(side="left")
        mark = tk.Canvas(brand, width=38, height=38, bg=BACKGROUND, highlightthickness=0)
        mark.pack(side="left", padx=(0, 11))
        mark.create_oval(2, 2, 36, 36, fill="#DDF7EF", outline="")
        mark.create_line(10, 25, 17, 17, 21, 21, 29, 11, fill=ACCENT, width=3, smooth=True, capstyle="round")
        tk.Label(brand, text="aircanvas", font=("Segoe UI", 17, "bold"), fg=INK, bg=BACKGROUND).pack(side="left")
        tk.Label(brand, text="STUDIO", font=("Segoe UI", 9, "bold"), fg=MUTED, bg=BACKGROUND).pack(side="left", padx=(10, 0), pady=(4, 0))

        self.camera_badge = tk.Label(
            header, text="●  CONNECTING CAMERA", font=("Segoe UI", 9, "bold"),
            fg="#B66A21", bg="#FFF4DF", padx=13, pady=8,
        )
        self.camera_badge.pack(side="right", padx=(12, 0))
        tk.Button(
            header, text="Save artwork  ↗", command=self.save_png,
            font=("Segoe UI", 10, "bold"), fg="white", bg=INK, activebackground="#2B3947",
            activeforeground="white", relief="flat", bd=0, padx=16, pady=10, cursor="hand2",
        ).pack(side="right")

        content = tk.Frame(shell, bg=BACKGROUND)
        content.grid(row=1, column=0, sticky="nsew")
        content.grid_columnconfigure(0, minsize=220)
        content.grid_columnconfigure(1, weight=1)
        content.grid_columnconfigure(2, minsize=315)
        content.grid_rowconfigure(0, weight=1)

        self._build_tools(content)
        self._build_canvas(content)
        self._build_camera_panel(content)

    def _card(self, parent: tk.Widget, **kwargs: object) -> tk.Frame:
        return tk.Frame(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1, **kwargs)

    def _section_label(self, parent: tk.Widget, text: str) -> None:
        tk.Label(parent, text=text.upper(), font=("Segoe UI", 8, "bold"), fg=MUTED, bg=PANEL).pack(anchor="w")

    def _build_tools(self, parent: tk.Frame) -> None:
        panel = self._card(parent)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        inner = tk.Frame(panel, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=18, pady=20)

        tk.Label(inner, text="Your tools", font=("Segoe UI", 15, "bold"), fg=INK, bg=PANEL).pack(anchor="w", pady=(0, 18))
        self._section_label(inner, "Tool")
        tool_row = tk.Frame(inner, bg=PANEL)
        tool_row.pack(fill="x", pady=(10, 22))
        self.draw_button = self._tool_button(tool_row, "✎", "Draw", "draw")
        self.draw_button.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.erase_button = self._tool_button(tool_row, "◌", "Erase", "erase")
        self.erase_button.pack(side="left", fill="x", expand=True, padx=(6, 0))

        self._section_label(inner, "Color")
        palette_frame = tk.Frame(inner, bg=PANEL)
        palette_frame.pack(anchor="w", pady=(12, 22))
        for index, color in enumerate(PALETTE):
            swatch = tk.Button(
                palette_frame, text="", command=lambda chosen=color: self._select_color(chosen),
                bg=color, activebackground=color, relief="flat", bd=0, width=2, height=1,
                cursor="hand2", highlightthickness=2, highlightbackground=PANEL,
            )
            swatch.grid(row=index // 3, column=index % 3, padx=(0, 10), pady=(0, 10), ipadx=3, ipady=5)
            self._swatch_buttons.append((swatch, color))

        self._section_label(inner, "Brush size")
        size_row = tk.Frame(inner, bg=PANEL)
        size_row.pack(fill="x", pady=(10, 0))
        self.size_value = tk.Label(size_row, text="8 px", font=("Segoe UI", 9, "bold"), fg=INK, bg=PANEL)
        self.size_value.pack(side="right")
        slider = ttk.Scale(inner, from_=3, to=28, orient="horizontal", command=self._set_brush_size)
        slider.set(8)
        slider.pack(fill="x", pady=(7, 4))
        tk.Label(inner, text="Fine                               Bold", font=("Segoe UI", 8), fg=MUTED, bg=PANEL).pack(fill="x")

        separator = tk.Frame(inner, bg=LINE, height=1)
        separator.pack(fill="x", pady=(24, 18))
        self._section_label(inner, "Canvas")
        actions = tk.Frame(inner, bg=PANEL)
        actions.pack(fill="x", pady=(10, 0))
        tk.Button(
            actions, text="↶  Undo stroke", command=self.undo, anchor="w", font=("Segoe UI", 9, "bold"),
            fg=INK, bg="#F3F6F8", activebackground="#E9EFF2", relief="flat", bd=0,
            padx=12, pady=10, cursor="hand2",
        ).pack(fill="x", pady=(0, 8))
        tk.Button(
            actions, text="⌫  Clear canvas", command=self.clear, anchor="w", font=("Segoe UI", 9, "bold"),
            fg="#A44C48", bg="#FFF1EF", activebackground="#FBE3E0", relief="flat", bd=0,
            padx=12, pady=10, cursor="hand2",
        ).pack(fill="x")
        self._select_color(self.selected_color.get())
        self._refresh_tool_buttons()

    def _tool_button(self, parent: tk.Widget, icon: str, label: str, mode: str) -> tk.Button:
        return tk.Button(
            parent, text=f"{icon}\n{label}", command=lambda: self._select_mode(mode),
            font=("Segoe UI", 9, "bold"), fg=INK, bg="#F3F6F8", activebackground="#E9EFF2",
            relief="flat", bd=0, padx=9, pady=10, cursor="hand2",
        )

    def _build_canvas(self, parent: tk.Frame) -> None:
        area = tk.Frame(parent, bg=BACKGROUND)
        area.grid(row=0, column=1, sticky="nsew", padx=(0, 16))
        area.grid_columnconfigure(0, weight=1)
        area.grid_rowconfigure(1, weight=1)
        title_row = tk.Frame(area, bg=BACKGROUND)
        title_row.grid(row=0, column=0, sticky="ew", pady=(2, 12))
        tk.Label(title_row, text="Drawing board", font=("Segoe UI", 15, "bold"), fg=INK, bg=BACKGROUND).pack(side="left")
        tk.Label(title_row, text="Untitled artwork", font=("Segoe UI", 9), fg=MUTED, bg=BACKGROUND).pack(side="right")

        canvas_card = self._card(area)
        canvas_card.grid(row=1, column=0, sticky="nsew")
        canvas_wrap = tk.Frame(canvas_card, bg=BACKGROUND, padx=8, pady=8)
        canvas_wrap.pack(fill="both", expand=True)
        self.canvas = Canvas(canvas_wrap, bg=BACKGROUND, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.cursor = self.canvas.create_oval(-20, -20, -10, -10, outline=ACCENT, width=2, fill="")

        hint = tk.Frame(area, bg=BACKGROUND)
        hint.grid(row=2, column=0, sticky="ew", pady=(11, 0))
        tk.Label(hint, text="✦  Tip: keep your hand inside the camera frame for the smoothest strokes.",
                 font=("Segoe UI", 9), fg=MUTED, bg=BACKGROUND).pack(side="left")
        tk.Label(hint, text="Ctrl + Z  Undo", font=("Segoe UI", 9, "bold"), fg=MUTED, bg=BACKGROUND).pack(side="right")

    def _build_camera_panel(self, parent: tk.Frame) -> None:
        panel = self._card(parent)
        panel.grid(row=0, column=2, sticky="nsew")
        inner = tk.Frame(panel, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=16, pady=18)
        title_row = tk.Frame(inner, bg=PANEL)
        title_row.pack(fill="x", pady=(0, 12))
        tk.Label(title_row, text="Live camera", font=("Segoe UI", 13, "bold"), fg=INK, bg=PANEL).pack(side="left")
        self.fps_label = tk.Label(title_row, textvariable=self.fps_text, font=("Segoe UI", 8, "bold"), fg=MUTED, bg="#F2F5F7", padx=8, pady=5)
        self.fps_label.pack(side="right")

        preview_shell = tk.Frame(inner, bg="#101820", height=222)
        preview_shell.pack(fill="x")
        preview_shell.pack_propagate(False)
        self.preview = tk.Label(preview_shell, bg="#101820", fg="#9DAAB6", text="Waiting for webcam…", font=("Segoe UI", 10))
        self.preview.pack(fill="both", expand=True)

        self.tracking_status = tk.Label(
            inner, textvariable=self.status_text, font=("Segoe UI", 9, "bold"), fg="#177E68",
            bg="#E8F8F3", anchor="w", padx=10, pady=9,
        )
        self.tracking_status.pack(fill="x", pady=(11, 18))
        self._section_label(inner, "How to draw")
        steps = [
            ("01", "Show your hand", "Keep one hand inside the frame."),
            ("02", "Move your index", "Place the cursor around the board."),
            ("03", "Pinch to draw", "Touch thumb and index to start a stroke."),
        ]
        for number, heading, detail in steps:
            row = tk.Frame(inner, bg=PANEL)
            row.pack(fill="x", pady=(13, 0))
            tk.Label(row, text=number, font=("Segoe UI", 8, "bold"), fg=ACCENT, bg="#E8F8F3", padx=7, pady=5).pack(side="left", anchor="n", padx=(0, 10))
            copy = tk.Frame(row, bg=PANEL)
            copy.pack(side="left", fill="x", expand=True)
            tk.Label(copy, text=heading, font=("Segoe UI", 9, "bold"), fg=INK, bg=PANEL, anchor="w").pack(fill="x")
            tk.Label(copy, text=detail, font=("Segoe UI", 8), fg=MUTED, bg=PANEL, anchor="w", wraplength=210, justify="left").pack(fill="x", pady=(3, 0))

        separator = tk.Frame(inner, bg=LINE, height=1)
        separator.pack(fill="x", pady=(18, 14))
        tk.Label(inner, text="SPACE  Pause drawing", font=("Segoe UI", 8, "bold"), fg=MUTED, bg=PANEL, anchor="w").pack(fill="x")
        tk.Label(inner, text="ESC  Quit app", font=("Segoe UI", 8, "bold"), fg=MUTED, bg=PANEL, anchor="w").pack(fill="x", pady=(7, 0))
        self.root.bind("<space>", self._space_pause)
        self.root.bind("<Escape>", lambda _event: self.close())

    def _set_brush_size(self, value: str) -> None:
        size = int(float(value))
        self.brush_size.set(size)
        self.size_value.configure(text=f"{size} px")

    def _select_color(self, color: str) -> None:
        self.selected_color.set(color)
        for button, swatch_color in self._swatch_buttons:
            button.configure(highlightbackground=ACCENT if color == swatch_color else PANEL)

    def _select_mode(self, mode: str) -> None:
        self.mode.set(mode)
        self.active_stroke = None
        self.previous_canvas_point = None
        self._refresh_tool_buttons()

    def _refresh_tool_buttons(self) -> None:
        for button, mode in ((self.draw_button, "draw"), (self.erase_button, "erase")):
            selected = self.mode.get() == mode
            button.configure(
                bg="#E4F6F0" if selected else "#F3F6F8",
                fg=ACCENT if selected else INK,
                activebackground="#E4F6F0" if selected else "#E9EFF2",
            )

    def _space_pause(self, _event: object) -> str:
        self.mode.set("pause" if self.mode.get() != "pause" else "draw")
        self.active_stroke = None
        self.previous_canvas_point = None
        self.draw_button.configure(bg="#F3F6F8", fg=INK)
        self.erase_button.configure(bg="#F3F6F8", fg=INK)
        self.status_text.set("Drawing paused · press Space to resume") if self.mode.get() == "pause" else None
        if self.mode.get() == "draw":
            self._refresh_tool_buttons()
        return "break"

    def _update(self) -> None:
        snapshot = self.camera.snapshot()
        self.last_snapshot = snapshot
        self.status_text.set(
            "Drawing paused · press Space to resume" if self.mode.get() == "pause" else snapshot.message
        )
        self.fps_text.set(f"{snapshot.fps:.0f} FPS" if snapshot.fps else "-- FPS")
        if snapshot.frame is not None:
            self.camera_badge.configure(text="●  CAMERA CONNECTED", fg="#177E68", bg="#E8F8F3")
        elif snapshot.message.startswith("Camera unavailable"):
            self.camera_badge.configure(text="●  CAMERA OFFLINE", fg="#A44C48", bg="#FFF1EF")
        else:
            self.camera_badge.configure(text="●  CONNECTING CAMERA", fg="#B66A21", bg="#FFF4DF")

        if snapshot.frame is not None:
            import cv2

            rgb = cv2.cvtColor(snapshot.frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            preview_width = max(self.preview.winfo_width(), 260)
            preview_height = max(self.preview.winfo_height(), 180)
            image.thumbnail((preview_width, preview_height), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.preview_photo, text="")

        self._update_canvas_pointer(snapshot)
        self.root.after(30, self._update)

    def _update_canvas_pointer(self, snapshot: TrackingSnapshot) -> None:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        if not snapshot.hand_found or snapshot.point is None:
            self.filtered_point = None
            self.previous_canvas_point = None
            self.active_stroke = None
            self.canvas.coords(self.cursor, -20, -20, -10, -10)
            return

        raw_x = min(max(snapshot.point[0], 0.015), 0.985) * width
        raw_y = min(max(snapshot.point[1], 0.015), 0.985) * height
        if self.filtered_point is None:
            self.filtered_point = (raw_x, raw_y)
        else:
            old_x, old_y = self.filtered_point
            self.filtered_point = (old_x * 0.48 + raw_x * 0.52, old_y * 0.48 + raw_y * 0.52)
        x, y = self.filtered_point
        if snapshot.drawing and self.mode.get() in ("draw", "erase"):
            self._paint_to(x, y)
            self.canvas.itemconfigure(self.cursor, outline="#FFFFFF", fill=ACCENT)
        else:
            self.active_stroke = None
            self.previous_canvas_point = None
            self.canvas.itemconfigure(self.cursor, outline=ACCENT, fill="")
        radius = max(6, self.brush_size.get() / 2)
        self.canvas.coords(self.cursor, x - radius, y - radius, x + radius, y + radius)
        self.canvas.tag_raise(self.cursor)

    def _paint_to(self, x: float, y: float) -> None:
        current = (x, y)
        color = BACKGROUND if self.mode.get() == "erase" else self.selected_color.get()
        width = max(3, self.brush_size.get() * (2.5 if self.mode.get() == "erase" else 1.0))
        if self.active_stroke is None:
            self.active_stroke = {"color": color, "width": width, "points": [current], "items": []}
            self.stroke_records.append(self.active_stroke)
            self.canvas.create_oval(
                x - width / 2, y - width / 2, x + width / 2, y + width / 2,
                fill=color, outline=color, tags=("stroke",),
            )
        elif self.previous_canvas_point is not None:
            old_x, old_y = self.previous_canvas_point
            item = self.canvas.create_line(
                old_x, old_y, x, y, fill=color, width=width, smooth=True,
                capstyle="round", joinstyle="round", tags=("stroke",),
            )
            self.active_stroke["items"].append(item)  # type: ignore[union-attr]
            self.active_stroke["points"].append(current)  # type: ignore[union-attr]
        self.previous_canvas_point = current

    def undo(self) -> None:
        self.active_stroke = None
        self.previous_canvas_point = None
        if not self.stroke_records:
            return
        stroke = self.stroke_records.pop()
        # Stroke items share a tag. Rebuild the board without the last stroke to keep undo simple and reliable.
        self.canvas.delete("stroke")
        for saved_stroke in self.stroke_records:
            self._redraw_stroke(saved_stroke)
        self.canvas.tag_raise(self.cursor)

    def _redraw_stroke(self, stroke: dict[str, object]) -> None:
        points = stroke["points"]
        color = str(stroke["color"])
        width = float(stroke["width"])
        if not isinstance(points, list) or not points:
            return
        first_x, first_y = points[0]
        self.canvas.create_oval(
            first_x - width / 2, first_y - width / 2, first_x + width / 2, first_y + width / 2,
            fill=color, outline=color, tags=("stroke",),
        )
        for start, end in zip(points, points[1:]):
            self.canvas.create_line(
                start[0], start[1], end[0], end[1], fill=color, width=width, smooth=True,
                capstyle="round", joinstyle="round", tags=("stroke",),
            )

    def clear(self) -> None:
        self.stroke_records.clear()
        self.active_stroke = None
        self.previous_canvas_point = None
        self.canvas.delete("stroke")

    def save_png(self) -> None:
        if not self.stroke_records:
            messagebox.showinfo("Nothing to save yet", "Draw something on the canvas first.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save your artwork", defaultextension=".png",
            filetypes=[("PNG image", "*.png")], initialfile="air-canvas-artwork.png",
        )
        if not path:
            return
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        image = Image.new("RGB", (width, height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        for stroke in self.stroke_records:
            points = stroke["points"]
            color = str(stroke["color"])
            line_width = max(1, int(float(stroke["width"])))
            if not isinstance(points, list) or not points:
                continue
            draw.line(points, fill=color, width=line_width, joint="curve")
            x, y = points[0]
            radius = line_width / 2
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        image.save(path, format="PNG")
        self.status_text.set(f"Artwork saved · {Path(path).name}")

    def close(self) -> None:
        self.camera.stop()
        self.root.after(100, self.root.destroy)


def main() -> None:
    root = Tk()
    AirCanvasApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

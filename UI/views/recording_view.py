import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from camera.camera import Camera


BACKGROUND = "#0C1B24"
SURFACE = "#142A35"
TEXT = "#F2F8F8"
MUTED = "#A8BDBF"


class RecordingView(tk.Frame):
    """Camera preview and feature-sample recording workspace."""

    def __init__(self, parent, on_back):
        super().__init__(parent, background=BACKGROUND)
        self.on_back = on_back
        self.sample_count = tk.StringVar(value="0")
        self.gesture_name = tk.StringVar()
        self.data_file = tk.StringVar()
        self.status = tk.StringVar(value="Camera preview stopped")
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.recording_event = threading.Event()
        self.recording_config = None
        self.recording_id = 0
        self.config_lock = threading.Lock()
        self.worker = None
        self.camera_image = None
        self.camera_canvas = None
        self._build()

    def _build(self):
        header = tk.Frame(self, background=BACKGROUND)
        header.pack(fill="x", padx=42, pady=(30, 0))
        tk.Button(header, text="←  Back", command=self._go_back, cursor="hand2",
                  font=("Helvetica", 14, "bold"), foreground="#12343B",
                  background="#CBE5EB", activebackground="#AFD6DF",
                  activeforeground="#102D34", relief="flat", borderwidth=0,
                  highlightthickness=0, padx=15, pady=10).pack(side="left", anchor="n")
        heading = tk.Frame(header, background=BACKGROUND)
        heading.pack(side="left", padx=(22, 0))
        tk.Label(heading, text="Record a gesture", foreground=TEXT, background=BACKGROUND,
                 font=("Helvetica", 26, "bold")).pack(anchor="w")
        tk.Label(heading, text="Name the gesture, then use the recording controls below.",
                 foreground=MUTED, background=BACKGROUND, font=("Helvetica", 14)).pack(anchor="w", pady=(3, 0))

        body = tk.Frame(self, background=BACKGROUND)
        body.pack(fill="both", expand=True, padx=42, pady=(34, 36))
        self.camera_canvas = tk.Canvas(body, background="#10232E", highlightbackground="#2B4A55",
                                       highlightthickness=1)
        self.camera_canvas.pack(fill="both", expand=True)
        self.camera_canvas.bind("<Configure>", self._draw_camera_placeholder)

        details = tk.Frame(body, background=BACKGROUND)
        details.pack(fill="x", pady=(18, 14))

        file_box = tk.Frame(details, background=SURFACE, padx=16, pady=12,
                            highlightbackground="#28424D", highlightthickness=1)
        file_box.pack(side="left", fill="x", expand=True, padx=(0, 7))
        tk.Label(file_box, text="DATA FILE", foreground="#8FA9AC", background=SURFACE,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.file_selector = ttk.Combobox(file_box, textvariable=self.data_file,
                                          values=self._data_files(), state="readonly",
                                          font=("Helvetica", 14))
        self.file_selector.pack(fill="x", pady=(5, 0))
        if self.file_selector["values"]:
            self.file_selector.current(0)

        name_box = tk.Frame(details, background=SURFACE, padx=16, pady=12,
                            highlightbackground="#28424D", highlightthickness=1)
        name_box.pack(side="left", fill="x", expand=True, padx=7)
        tk.Label(name_box, text="GESTURE NAME", foreground="#8FA9AC", background=SURFACE,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Entry(name_box, textvariable=self.gesture_name, foreground=TEXT, insertbackground=TEXT,
                 background=SURFACE, relief="flat", font=("Helvetica", 17)).pack(fill="x", pady=(4, 0))
        count_box = tk.Frame(details, background=SURFACE, padx=20, pady=12,
                             highlightbackground="#28424D", highlightthickness=1)
        count_box.pack(side="left", padx=(7, 0))
        tk.Label(count_box, text="SAMPLES", foreground="#8FA9AC", background=SURFACE,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Label(count_box, textvariable=self.sample_count, foreground="#64B9B0", background=SURFACE,
                 font=("Helvetica", 24, "bold")).pack(anchor="w", pady=(3, 0))

        controls = tk.Frame(body, background=BACKGROUND)
        controls.pack(fill="x")
        tk.Label(controls, textvariable=self.status, foreground="#B6CED0", background=BACKGROUND,
                 font=("Helvetica", 13)).pack(side="left")
        tk.Button(controls, text="Stop", command=self.stop_recording, cursor="hand2",
                  font=("Helvetica", 14, "bold"), foreground="#12343B", background="#CBE5EB",
                  activebackground="#AFD6DF", relief="flat", borderwidth=0, highlightthickness=0,
                  padx=26, pady=11).pack(side="right")
        tk.Button(controls, text="Start", command=self.start_recording, cursor="hand2",
                  font=("Helvetica", 14, "bold"), foreground="#12343B", background="#C9F0EB",
                  activebackground="#A8DED8", relief="flat", borderwidth=0, highlightthickness=0,
                  padx=26, pady=11).pack(side="right", padx=(0, 10))

    def start_preview(self):
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.status.set("Starting camera preview…")
        self.worker = threading.Thread(target=self._preview_loop, daemon=True)
        self.worker.start()
        self.after(20, self._read_events)

    def stop_preview(self):
        self.recording_event.clear()
        self.stop_event.set()

    def start_recording(self):
        label = self.gesture_name.get().strip()
        filename = self.data_file.get().strip()
        if not label:
            self.status.set("Enter a gesture name before starting")
            return
        if not filename:
            self.status.set("Choose a CSV file from the data folder")
            return

        path = Path("data") / filename
        if path.suffix.lower() != ".csv" or not path.is_file():
            self.status.set("Choose a valid CSV file from the data folder")
            return

        with self.config_lock:
            self.recording_id += 1
            self.recording_config = {
                "label": label,
                "filename": str(path),
                "session": self.recording_id,
            }
        self.sample_count.set("0")
        self.recording_event.set()
        self.status.set(f"Recording {label} into {filename}")

    def stop_recording(self):
        self.recording_event.clear()
        self.status.set("Recording stopped")

    def _go_back(self):
        self.stop_preview()
        self.on_back()

    def _preview_loop(self):
        camera = None
        hand_detector = None
        feature_extractor = None
        collector = None
        active_session = None
        frame_counter = 0
        sample_count = 0
        try:
            camera = Camera()
            self.events.put(("status", "Camera preview active"))
            while not self.stop_event.is_set():
                frame = camera.read()
                if frame is None:
                    self.events.put(("status", "Camera unavailable — check camera permissions"))
                    break
                if self.events.qsize() < 2:
                    self.events.put(("frame", frame.copy()))

                if self.recording_event.is_set():
                    with self.config_lock:
                        config = self.recording_config.copy() if self.recording_config else None
                    if config is not None:
                        if active_session != config["session"]:
                            from data_collection.data_collector import DataCollector
                            from detection.hand_detector import HandDetector
                            from feature_extraction.feature_extractor import FeatureExtractor

                            if hand_detector is None:
                                hand_detector = HandDetector()
                                feature_extractor = FeatureExtractor()
                            collector = DataCollector(config["filename"])
                            active_session = config["session"]
                            frame_counter = 0
                            sample_count = 0

                        result = hand_detector.detect(frame)
                        if result.hand_landmarks:
                            points = hand_detector.get_landmark_points(result)
                            frame_counter += 1
                            if frame_counter % 7 == 0:
                                features = feature_extractor.extract_features(points)
                                hand = self._handedness(result)
                                collector.save_sample(config["label"], hand, features)
                                sample_count += 1
                                self.events.put(("sample", sample_count))
                                frame_counter = 0
                time.sleep(0.03)
        except Exception as error:
            self.events.put(("status", f"Preview unavailable: {error}"))
        finally:
            if camera is not None:
                camera.release()

    def _read_events(self):
        while True:
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "status":
                self.status.set(payload)
            elif event == "frame":
                self._show_frame(payload)
            elif event == "sample":
                self.sample_count.set(str(payload))
        if self.worker and self.worker.is_alive():
            self.after(20, self._read_events)

    def _show_frame(self, frame):
        width, height = self.camera_canvas.winfo_width(), self.camera_canvas.winfo_height()
        if width < 2 or height < 2:
            return
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.camera_image = ImageTk.PhotoImage(image)
        self.camera_canvas.delete("placeholder")
        self.camera_canvas.delete("frame")
        self.camera_canvas.create_image(width / 2, height / 2, image=self.camera_image, tags="frame")

    def _draw_camera_placeholder(self, event):
        if self.camera_image is not None:
            return
        canvas = event.widget
        canvas.delete("placeholder")
        x, y = event.width / 2, event.height / 2
        canvas.create_rectangle(x - 24, y - 19, x + 24, y + 19,
                                outline="#759CA0", width=2, tags="placeholder")
        canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#759CA0",
                           outline="", tags="placeholder")
        canvas.create_text(x, y + 54, text="CAMERA FEED", fill="#89A6A8",
                           font=("Helvetica", 12, "bold"), tags="placeholder")

    @staticmethod
    def _data_files():
        return sorted(path.name for path in Path("data").glob("*.csv"))

    @staticmethod
    def _handedness(result):
        try:
            return result.handedness[0][0].category_name.lower()
        except (AttributeError, IndexError):
            return "unknown"

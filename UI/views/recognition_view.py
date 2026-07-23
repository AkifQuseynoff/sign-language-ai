import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
from PIL import Image, ImageTk


BACKGROUND = "#0C1B24"
SURFACE = "#142A35"
TEXT = "#F2F8F8"
MUTED = "#A8BDBF"


class StatCard(tk.Frame):
    def __init__(self, parent, label, value, accent=False):
        super().__init__(parent, background=SURFACE, padx=20, pady=16,
                         highlightbackground="#28424D", highlightthickness=1)
        tk.Label(self, text=label.upper(), foreground="#8FA9AC", background=SURFACE,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Label(self, textvariable=value, foreground="#64B9B0" if accent else TEXT,
                 background=SURFACE, font=("Helvetica", 25 if accent else 23, "bold")).pack(
                     anchor="w", pady=(4, 0))


class RecognitionView(tk.Frame):
    def __init__(self, parent, on_back):
        super().__init__(parent, background=BACKGROUND)
        self.on_back = on_back
        self.prediction = tk.StringVar(value="—")
        self.confidence = tk.StringVar(value="— %")
        self.status = tk.StringVar(value="Waiting for camera")
        self.model_name = tk.StringVar()
        self.events = queue.Queue()
        self.stop_event = threading.Event()
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
        tk.Label(heading, text="Live recognition", foreground=TEXT, background=BACKGROUND,
                 font=("Helvetica", 26, "bold")).pack(anchor="w")
        tk.Label(heading, text="Position your hand inside the frame to begin translating.",
                 foreground=MUTED, background=BACKGROUND, font=("Helvetica", 14)).pack(anchor="w", pady=(3, 0))

        body = tk.Frame(self, background=BACKGROUND)
        body.pack(fill="both", expand=True, padx=42, pady=(34, 36))
        self.camera_canvas = tk.Canvas(body, background="#10232E", highlightbackground="#2B4A55",
                                       highlightthickness=1)
        self.camera_canvas.pack(fill="both", expand=True)
        self.camera_canvas.bind("<Configure>", self._draw_camera_placeholder)

        model_box = tk.Frame(body, background=SURFACE, padx=16, pady=11,
                             highlightbackground="#28424D", highlightthickness=1)
        model_box.pack(fill="x", pady=(14, 0))
        tk.Label(model_box, text="RECOGNITION MODEL", foreground="#8FA9AC", background=SURFACE,
                 font=("Helvetica", 11, "bold")).pack(side="left")
        self.model_selector = ttk.Combobox(model_box, textvariable=self.model_name,
                                           values=self._model_files(), state="readonly",
                                           width=28, font=("Helvetica", 13))
        self.model_selector.pack(side="right")
        if self.model_selector["values"]:
            default = "letters_model_v1.pkl"
            self.model_name.set(default if default in self.model_selector["values"] else self.model_selector["values"][0])
        self.model_selector.bind("<<ComboboxSelected>>", self._reload_selected_model)

        status = tk.Frame(body, background=BACKGROUND)
        status.pack(fill="x", pady=(14, 18))
        tk.Label(status, text="●", foreground="#58AFA5", background=BACKGROUND,
                 font=("Helvetica", 14)).pack(side="left")
        tk.Label(status, textvariable=self.status, foreground="#B6CED0", background=BACKGROUND,
                 font=("Helvetica", 13)).pack(side="left", padx=(6, 0))

        metrics = tk.Frame(body, background=BACKGROUND)
        metrics.pack(fill="x")
        StatCard(metrics, "Prediction", self.prediction, accent=True).pack(
            side="left", fill="both", expand=True, padx=(0, 7))
        StatCard(metrics, "Confidence", self.confidence).pack(
            side="left", fill="both", expand=True, padx=(7, 0))

    def start_recognition(self):
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        model_path = Path("models") / self.model_name.get()
        if not model_path.is_file():
            self.status.set("Choose a valid model from the models folder")
            return
        self.status.set(f"Loading {model_path.name}…")
        self.worker = threading.Thread(target=self._recognition_loop,
                                       args=(str(model_path),), daemon=True)
        self.worker.start()
        self.after(20, self._read_events)

    def stop_recognition(self):
        self.stop_event.set()
        self.status.set("Camera stopped")

    def _go_back(self):
        self.stop_recognition()
        self.on_back()

    def _reload_selected_model(self, event):
        del event
        self.status.set("Switching recognition model…")
        self.stop_event.set()
        self.after(50, self._start_when_stopped)

    def _start_when_stopped(self):
        if self.worker and self.worker.is_alive():
            self.after(50, self._start_when_stopped)
            return
        self.start_recognition()

    def _recognition_loop(self, model_path):
        pipeline = None
        last_value = None
        last_frame_time = 0.0
        try:
            from main import RecognitionPipeline
            pipeline = RecognitionPipeline(model_path)
            self.events.put(("status", f"Camera active — {Path(model_path).name}"))
            while not self.stop_event.is_set():
                detection_result, prediction, confidence = pipeline.process_next_frame()
                if detection_result is None:
                    self.events.put(("status", "Camera unavailable — check camera permissions"))
                    break
                now = time.monotonic()
                if now - last_frame_time >= 0.02 and self.events.qsize() < 3:
                    self.events.put(("frame", pipeline.last_frame.copy()))
                    last_frame_time = now
                if prediction is not None:
                    value = (prediction, confidence)
                    if value != last_value:
                        self.events.put(("prediction", value))
                        print(f"Prediction: {prediction} | Confidence: {confidence:.1f}%")
                        last_value = value
        except Exception as error:
            self.events.put(("status", f"Recognition unavailable: {error}"))
        finally:
            if pipeline is not None:
                pipeline.close()

    def _read_events(self):
        while True:
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "status":
                self.status.set(payload)
            elif event == "prediction":
                prediction, confidence = payload
                self.prediction.set(str(prediction))
                self.confidence.set(f"{confidence:.1f} %")
            elif event == "frame":
                self._show_frame(payload)

        if self.worker and self.worker.is_alive():
            self.after(20, self._read_events)

    def _show_frame(self, frame):
        canvas = self.camera_canvas
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 2 or height < 2:
            return
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.camera_image = ImageTk.PhotoImage(image)
        canvas.delete("placeholder")
        canvas.delete("frame")
        canvas.create_image(width / 2, height / 2, image=self.camera_image, tags="frame")

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
    def _model_files():
        return sorted(path.name for path in Path("models").glob("*.pkl"))

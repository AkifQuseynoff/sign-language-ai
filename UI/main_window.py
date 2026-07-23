import tkinter as tk

from UI.views.home_view import HomeView
from UI.views.recognition_view import RecognitionView
from UI.views.recording_view import RecordingView


class MainWindow:
    """The application shell, implemented with Python's built-in Tkinter UI."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ASL Translator")
        self.root.geometry("1280x800")
        self.root.minsize(900, 620)
        self.root.configure(background="#0C1B24")

        self.content = tk.Frame(self.root, background="#0C1B24")
        self.content.pack(fill="both", expand=True)

        self.home_view = HomeView(
            self.content,
            on_recognition=self.show_recognition,
            on_recording=self.show_recording,
        )
        self.recognition_view = RecognitionView(self.content, on_back=self.show_home)
        self.recording_view = RecordingView(self.content, on_back=self.show_home)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.show_home()

    def _show(self, view):
        for child in self.content.winfo_children():
            child.pack_forget()
        view.pack(fill="both", expand=True)

    def show_home(self):
        self.recognition_view.stop_recognition()
        self.recording_view.stop_preview()
        self._show(self.home_view)

    def show_recognition(self):
        self._show(self.recognition_view)
        self.recognition_view.start_recognition()

    def show_recording(self):
        self._show(self.recording_view)
        self.recording_view.start_preview()

    def run(self):
        self.root.mainloop()

    def _close(self):
        self.recognition_view.stop_recognition()
        self.recording_view.stop_preview()
        self.root.destroy()

import tkinter as tk


BACKGROUND = "#0C1B24"
SURFACE = "#142A35"
TEXT = "#F2F8F8"
MUTED = "#A8BDBF"


class HomeView(tk.Frame):
    def __init__(self, parent, on_recognition, on_recording):
        super().__init__(parent, background=BACKGROUND)
        self.on_recognition = on_recognition
        self.on_recording = on_recording
        self._build()

    @staticmethod
    def _button(parent, text, command, primary=False):
        # Light button surfaces give both labels reliable contrast on macOS Tk.
        color = "#C9F0EB" if primary else "#CBE5EB"
        active = "#A8DED8" if primary else "#AFD6DF"
        return tk.Button(
            parent, text=text, command=command, cursor="hand2",
            font=("Helvetica", 15, "bold"), foreground="#12343B",
            background=color, activebackground=active, activeforeground="#102D34",
            relief="flat", borderwidth=0, highlightthickness=0, padx=28, pady=14,
        )

    def _build(self):
        header = tk.Frame(self, background=BACKGROUND)
        header.pack(fill="x", padx=42, pady=(30, 0))
        tk.Label(header, text="ASL", foreground=TEXT, background=BACKGROUND,
                 font=("Helvetica", 16, "bold")).pack(side="left")
        tk.Label(header, text=" TRANSLATOR", foreground="#67B9B1", background=BACKGROUND,
                 font=("Helvetica", 16, "bold")).pack(side="left")

        hero = tk.Frame(self, background=BACKGROUND)
        hero.place(relx=.5, rely=.54, anchor="center")
        tk.Label(hero, text="Sign language,\nmade visible.", justify="center",
                 foreground=TEXT, background=BACKGROUND,
                 font=("Helvetica", 42, "bold")).pack()
        tk.Label(hero,
                 text="Understand signs in real time with a simple, private\nexperience built around your camera.",
                 justify="center", foreground=MUTED, background=BACKGROUND,
                 font=("Helvetica", 16)).pack(pady=(16, 38))

        actions = tk.Frame(hero, background=SURFACE, padx=22, pady=22)
        actions.pack()
        self._button(actions, "Recognition", self.on_recognition, primary=True).pack(
            fill="x", pady=(0, 12))
        self._button(actions, "Recording", self.on_recording).pack(fill="x")

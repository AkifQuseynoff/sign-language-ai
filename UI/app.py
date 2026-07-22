import customtkinter as ctk


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ASL Gesture Translator")
        self.geometry("900x700")

        self.title_label = ctk.CTkLabel(
            self,
            text="ASL Gesture Translator",
            font=("Arial", 32)
        )

        self.title_label.pack(
            pady=30
        )

        self.status_label = ctk.CTkLabel(
            self,
            text="Ready",
            font=("Arial", 20)
        )

        self.status_label.pack(
            pady=20
        )


if __name__ == "__main__":
    app = App()
    app.mainloop()
import customtkinter as ctk
import cv2

from PIL import Image, ImageTk

from camera.camera import Camera

def resize_image(image, max_width, max_height):
    width, height = image.size

    ratio = min(
        max_width / width,
        max_height / height
    )

    new_width = int(width * ratio)
    new_height = int(height * ratio)

    return image.resize(
        (new_width, new_height)
    )


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ASL Gesture Translator")
        self.geometry("900x700")

        self.camera = Camera()

        self.title_label = ctk.CTkLabel(
            self,
            text="ASL Gesture Translator",
            font=("Arial", 32))
        self.title_label.pack(pady=20)

        self.camera_label = ctk.CTkLabel(
        self,
        text="",
        width=700,
        height=450)
        self.camera_label.pack(pady=20,padx=20)


        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=20)

        self.recognize_button = ctk.CTkButton(
              self.button_frame,
              text="Recognize Gestures")

        self.recognize_button.pack(
            side="left",
            padx=10)

        self.record_button = ctk.CTkButton(
            self.button_frame,
            text="Record Gestures")

        self.record_button.pack(
            side="left",
            padx=10)

        self.update_camera()


    def update_camera(self):
        frame = self.camera.read()

        if frame is not None:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(frame)

            image = resize_image(
                image,
                700,
                450
            )

            photo = ImageTk.PhotoImage(
                image
            )

            self.camera_label.configure(
                image=photo
            )

            self.camera_label.image = photo

        self.after(
            30,
            self.update_camera
        )


    def close(self):
        self.camera.release()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol(
        "WM_DELETE_WINDOW",
        app.close
    )
    app.mainloop()
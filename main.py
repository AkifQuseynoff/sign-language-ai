from recognition.gesture_recognizer import GestureRecognizer
from camera.camera import Camera
from UI.display import Display
 
camera = Camera()
recognizer = GestureRecognizer()
display = Display()
prediction = ""

while True:
    frame = camera.read()
    if frame is None:
        break
    #recognize gesture 
    detection_result , new_prediction = recognizer.process_frame(frame)
    if new_prediction is not None:
        prediction = new_prediction

    # Draw landmarks
    display.show(frame, detection_result, prediction)

    if display.should_close():
        break

camera.release()
display.close()
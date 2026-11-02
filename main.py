"""Recognition entry point and reusable camera/classifier pipeline."""

from camera.camera import Camera
from recognition.gesture_recognizer import GestureRecognizer


class RecognitionPipeline:
    """Owns the camera and recognizer for either the desktop UI or CLI preview."""

    def __init__(self, model_path="models/letters_model_v2.pkl"):
        self.camera = Camera()
        self.recognizer = GestureRecognizer(model_path)
        self.prediction = None
        self.confidence = None
        self.last_frame = None

    def process_next_frame(self):
        frame = self.camera.read()
        if frame is None:
            return None, self.prediction, self.confidence

        self.last_frame = frame

        detection_result, result = self.recognizer.process_frame_with_confidence(frame)
        if result is None:
            # No hand in view, so there is nothing to show.
            self.prediction, self.confidence = None, None
        else:
            self.prediction, self.confidence = result

        return detection_result, self.prediction, self.confidence

    def close(self):
        self.camera.release()


def run_preview():
    """Keep the original OpenCV preview available when main.py is run directly."""
    from UI.display import Display

    pipeline = RecognitionPipeline()
    display = Display()
    try:
        while True:
            detection_result, prediction, _ = pipeline.process_next_frame()
            if detection_result is None:
                break
            display.show(pipeline.last_frame, detection_result, prediction or "")
            if display.should_close():
                break
    finally:
        pipeline.close()
        display.close()


if __name__ == "__main__":
    run_preview()

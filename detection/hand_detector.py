import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


class HandDetector:
    # The datasets were recorded with the right hand. MediaPipe names hands as
    # if the image were a mirrored selfie, but our camera image is not
    # mirrored, so a real right hand is reported as "Left".
    TRAINED_SIDE = "Left"

    def __init__(self, model_path="models/hand_landmarker.task"):
        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_id = 0

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        self.frame_id += 1
        timestamp_ms = self.frame_id * 33

        return self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

    def get_side(self, detection_result):
        """MediaPipe's name for the first hand: "Left", "Right" or None."""
        try:
            return detection_result.handedness[0][0].category_name
        except (AttributeError, IndexError):
            return None

    def get_landmark_points(self, detection_result):
        hand_landmarks= detection_result.hand_landmarks[0]
        points = []

        # The other hand is flipped left-to-right so it looks like the hand
        # the models were trained on.
        mirror = self.get_side(detection_result) not in (self.TRAINED_SIDE, None)

        for landmark in hand_landmarks:
            x = 1 - landmark.x if mirror else landmark.x
            points.append(
                (x, landmark.y, landmark.z)
            )

        return points
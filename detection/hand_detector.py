import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


class HandDetector:
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

    def get_landmark_points(self, detection_result):
        hand_landmarks= detection_result.hand_landmarks[0]
        points = []

        for landmark in hand_landmarks:
            points.append(
                (landmark.x, landmark.y, landmark.z)
            )

        return points
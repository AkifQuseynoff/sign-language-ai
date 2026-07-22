import cv2
import mediapipe as mp
import math
import numpy as np
from features.feature_extractor import FeatureExtractor
from data_collection.data_collector import DataCollector

from mediapipe.tasks import python
from mediapipe.tasks.python import vision



def get_landmark_points(hand_landmarks):
    points = []

    for landmark in hand_landmarks:
        points.append(
            (landmark.x, landmark.y, landmark.z)
        )

    return points

# Create hand detector
base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)
feature_extractor = FeatureExtractor()
collector = DataCollector("data/letters_v1.csv")

# Camera
cap = cv2.VideoCapture(0)
frame_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    result = detector.detect(mp_image)
    angle=0
    if result.hand_landmarks:
        
        hand = result.hand_landmarks[0]
        points = get_landmark_points(hand)
        features = feature_extractor.extract_features(points)
        frame_counter += 1
        if frame_counter % 7 == 0:
            collector.save_sample("Letter_Y","right",features)
            frame_counter = 0
    # Draw landmarks
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for landmark in hand:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()
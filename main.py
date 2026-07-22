import cv2
import mediapipe as mp
import math
import numpy as np
from features.feature_extractor import FeatureExtractor
from data_collection.data_collector import DataCollector
import joblib
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model=joblib.load("models/letters_model_v1.pkl")

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
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)
feature_extractor = FeatureExtractor()

# Camera
cap = cv2.VideoCapture(0)
frame_id = 0
prediction_counter = 0
prediction=""
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
    frame_id += 1
    timestamp_ms = frame_id * 33
    result = detector.detect_for_video(mp_image,timestamp_ms)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        points = get_landmark_points(hand)
        features = feature_extractor.extract_features(points)
        prediction_counter += 1
        if prediction_counter == 5:
            prediction_counter = 0
            prediction = model.predict([features])[0]
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
    cv2.putText(frame,prediction,(30,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),3)

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()
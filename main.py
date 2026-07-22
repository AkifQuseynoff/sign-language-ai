import cv2
import mediapipe as mp
import math
import numpy as np
from features.feature_extractor import FeatureExtractor
from data_collection.data_collector import DataCollector
import joblib
from camera.camera import Camera
from detection.hand_detector import HandDetector

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

hand_detector = HandDetector()

feature_extractor = FeatureExtractor()

# Camera
camera = Camera()
frame_id = 0
prediction_counter = 0
prediction=""
while True:
    frame = camera.read()

    if frame is None:
        break

    result = hand_detector.detect(frame)

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



camera.release()
cv2.destroyAllWindows()
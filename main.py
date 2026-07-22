from features.feature_extractor import FeatureExtractor
from camera.camera import Camera
from detection.hand_detector import HandDetector
from recognition.classifier import GestureClassifier
import cv2
from UI.display import Display

model= GestureClassifier()

def get_landmark_points(hand_landmarks):
    points = []

    for landmark in hand_landmarks:
        points.append(
            (landmark.x, landmark.y, landmark.z)
        )

    return points

hand_detector = HandDetector()
feature_extractor = FeatureExtractor()
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
            prediction = model.predict(features)
    # Draw landmarks
    Display().show(frame, result, prediction)

    if Display().should_close():
        break



camera.release()
Display().close()
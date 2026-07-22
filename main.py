from features.feature_extractor import FeatureExtractor
from camera.camera import Camera
from detection.hand_detector import HandDetector
from recognition.classifier import GestureClassifier
from UI.display import Display


classifier= GestureClassifier()
hand_detector = HandDetector()
feature_extractor = FeatureExtractor()
camera = Camera()
display = Display()
prediction = ""

while True:
    frame = camera.read()
    if frame is None:
        break

    detection_result = hand_detector.detect(frame)

    if detection_result.hand_landmarks:
        points = hand_detector.get_landmark_points(detection_result)
        features = feature_extractor.extract_features(points)
        new_prediction = classifier.update(features)

        if new_prediction is not None:
            prediction = new_prediction
    # Draw landmarks
    display.show(frame, detection_result, prediction)

    if display.should_close():
        break

camera.release()
display.close()
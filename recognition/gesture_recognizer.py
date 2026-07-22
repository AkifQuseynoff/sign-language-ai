from features.feature_extractor import FeatureExtractor
from detection.hand_detector import HandDetector
from recognition.classifier import GestureClassifier

class GestureRecognizer:
    def __init__(self):
        self.classifier = GestureClassifier()
        self.hand_detector = HandDetector()
        self.feature_extractor = FeatureExtractor()

    def process_frame(self, frame):
        new_prediction = None
        detection_result = self.hand_detector.detect(frame)
        
        if detection_result.hand_landmarks:
            points = self.hand_detector.get_landmark_points(detection_result)
            features = self.feature_extractor.extract_features(points)
            new_prediction = self.classifier.update(features)
            
        return detection_result, new_prediction
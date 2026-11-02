from feature_extraction.feature_extractor import FeatureExtractor
from detection.hand_detector import HandDetector
from recognition.classifier import GestureClassifier

class GestureRecognizer:
    def __init__(self, model_path="models/letters_model_v2.pkl"):
        self.classifier = GestureClassifier(model_path)
        self.hand_detector = HandDetector()
        self.feature_extractor = FeatureExtractor()

    def process_frame(self, frame):
        detection_result, result = self.process_frame_with_confidence(frame)
        return detection_result, result[0] if result else None

    def process_frame_with_confidence(self, frame):
        new_prediction = None
        detection_result = self.hand_detector.detect(frame)
        
        if detection_result.hand_landmarks:
            points = self.hand_detector.get_landmark_points(detection_result)
            features = self.feature_extractor.extract_features(points)
            new_prediction = self.classifier.update_with_confidence(features)
            
        return detection_result, new_prediction

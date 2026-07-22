from camera.camera import Camera
from detection.hand_detector import HandDetector
from feature_extraction.feature_extractor import FeatureExtractor
from data_collection.data_collector import DataCollector
from UI.display import Display


camera = Camera()
hand_detector = HandDetector()
feature_extractor = FeatureExtractor()
collector = DataCollector("data/letters_v1.csv")
display = Display()

frame_counter = 0

while True:
    frame = camera.read()

    if frame is None:
        break

    result = hand_detector.detect(frame)

    if result.hand_landmarks:
        points = hand_detector.get_landmark_points(result)
        features = feature_extractor.extract_features(points)

        frame_counter += 1

        if frame_counter % 7 == 0:
            collector.save_sample(
                "Letter_C",
                "right",
                features
            )

            frame_counter = 0

    display.show(frame, result, "")

    if display.should_close():
        break


camera.release()
display.close()
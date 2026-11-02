from collections import deque

import joblib
import numpy as np
class GestureClassifier:

    def __init__(self, model_path="models/letters_model_v2.pkl",
                 history_size=10, min_confidence=60.0):
        self.model = joblib.load(model_path)
        # The model's probabilities for the last few frames. A deque with
        # maxlen drops the oldest entry automatically when a new one is added.
        self.history = deque(maxlen=history_size)
        self.min_confidence = min_confidence

    def predict(self, features):
        return self.model.predict([features])[0]

    def predict_with_confidence(self, features):
        """Return the predicted class along with the model's certainty as a percent."""
        sample = np.array([features])
        prediction = self.model.predict(sample)[0]
        if not hasattr(self.model, "predict_proba"):
            return prediction, 100.0

        confidence = float(np.max(self.model.predict_proba(sample)[0]) * 100)
        return prediction, confidence

    def update(self, features):
        prediction, _ = self.update_with_confidence(features)
        return prediction

    def update_with_confidence(self, features):
        """Add one frame and return the smoothed (prediction, confidence).

        The probabilities of the last few frames are averaged, so one shaky
        frame can't change the answer. If even the best class is below
        min_confidence, the prediction is None, meaning "not sure".
        """
        if not hasattr(self.model, "predict_proba"):
            return self.predict_with_confidence(features)

        probabilities = self.model.predict_proba(np.array([features]))[0]
        self.history.append(probabilities)

        average = np.mean(self.history, axis=0)
        best = int(np.argmax(average))
        confidence = float(average[best] * 100)

        if confidence < self.min_confidence:
            return None, confidence
        return self.model.classes_[best], confidence

    def reset(self):
        """Forget earlier frames, for example when the hand leaves the camera."""
        self.history.clear()

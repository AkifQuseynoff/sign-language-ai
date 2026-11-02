import joblib
import numpy as np
class GestureClassifier:

    def __init__(self, model_path="models/letters_model_v2.pkl"):
        self.model = joblib.load(model_path)
        self.prediction_counter = 0

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
        result = self.update_with_confidence(features)
        return result[0] if result else None

    def update_with_confidence(self, features):
        self.prediction_counter += 1

        if self.prediction_counter < 5:
            return None

        self.prediction_counter = 0
        return self.predict_with_confidence(features)

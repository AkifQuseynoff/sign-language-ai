import joblib
import numpy as np
class GestureClassifier:

    def __init__(self, model_path="models/letters_model_v1.pkl"):
        self.model = joblib.load(model_path)
        self.prediction_counter = 0

    def predict(self, features):
        print(type(features))
        print(features)
        print("shape:", np.array(features).shape)

        return self.model.predict([features])[0]
    
    def update(self, features):
        self.prediction_counter += 1

        if self.prediction_counter < 5:
            return None

        self.prediction_counter = 0
        return self.predict(features)
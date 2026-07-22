import joblib
import numpy as np
class GestureClassifier:

    def __init__(self, model_path="models/letters_model_v1.pkl"):

        self.model = joblib.load(model_path)

    def predict(self, features):
        print(type(features))
        print(features)
        print("shape:", np.array(features).shape)

        return self.model.predict([features])[0]
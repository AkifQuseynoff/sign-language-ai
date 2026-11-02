"""Train a gesture classifier from a CSV dataset.

Run from the project root, for example:
    python training/train_model.py --data data/letters_v2.csv --output models/letters_model_v2.pkl
"""

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def split_by_recording_order(data, test_size=0.2):
    """Hold out the last part of each gesture's samples for testing.

    Samples are recorded a few frames apart, so neighbouring rows are almost
    identical. A random split would put near-copies of the same pose in both
    the training and test sets, and the accuracy would look better than it
    really is. Testing on the samples recorded last keeps the two sets apart.
    """
    label = data.iloc[:, 0]
    position = data.groupby(label).cumcount()
    total = label.map(label.value_counts())
    is_test = position >= total * (1 - test_size)
    return data[~is_test], data[is_test]


def train(data_path, output_path):
    data = pd.read_csv(data_path)

    train_data, test_data = split_by_recording_order(data)
    # .values gives plain numbers without column names, the same form the
    # app passes in when it predicts, so scikit-learn doesn't warn about it.
    X_train, y_train = train_data.iloc[:, 2:].values, train_data.iloc[:, 0]
    X_test, y_test = test_data.iloc[:, 2:].values, test_data.iloc[:, 0]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    # The test above measured how good the model is. Now train it again on
    # every sample so the saved model learns from all of the data.
    model.fit(data.iloc[:, 2:].values, data.iloc[:, 0])
    joblib.dump(model, output_path)
    print("Model saved to", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a gesture classifier.")
    parser.add_argument("--data", default="data/letters_v2.csv",
                        help="CSV dataset to train on")
    parser.add_argument("--output", default="models/letters_model_v2.pkl",
                        help="where to save the trained model")
    args = parser.parse_args()
    train(args.data, args.output)

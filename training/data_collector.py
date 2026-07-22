import csv
import os

class DataCollector:
    def __init__(self, filename):
        self.filename = filename
    def save_sample(self, label, hand, features):
        file_exists = (
            os.path.exists(self.filename)
            and os.path.getsize(self.filename) > 0
        )
        with open(
            self.filename,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            if not file_exists:
                header = [
                    "label",
                    "hand"
                ]
                header += [
                    f"feature_{i}"
                    for i in range(len(features))
                ]
                writer.writerow(header)
            writer.writerow(
                [label, hand] + features
            )
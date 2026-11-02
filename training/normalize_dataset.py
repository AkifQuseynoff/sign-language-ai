"""Convert a dataset recorded before palm-size normalization.

Older CSVs store bone lengths and distances in raw camera units. The feature
extractor now divides them by the palm size, so old rows must be converted
the same way before they can be mixed with new samples or used for training.

Run from the project root, for example:
    python training/normalize_dataset.py data/letters_v1.csv data/letters_v2.csv
"""

import argparse

import pandas as pd


# Feature layout: 10 angles, 20 bone lengths, 10 tip-to-tip distances,
# 5 wrist-to-tip distances, then 15 finger-direction numbers.
FIRST_LENGTH = 10
LAST_LENGTH = 44

# Bone (0, 9), wrist to middle finger knuckle, is the 9th bone, so it is
# stored in feature column 10 + 8 = 18. This is the palm size.
PALM_SIZE = 18


def normalize(input_path, output_path):
    data = pd.read_csv(input_path)
    features = data.iloc[:, 2:].copy()
    features.columns = [f"feature_{i}" for i in range(len(features.columns))]

    palm = features[f"feature_{PALM_SIZE}"].copy()
    for i in range(FIRST_LENGTH, LAST_LENGTH + 1):
        features[f"feature_{i}"] = features[f"feature_{i}"] / palm

    output = pd.concat([data.iloc[:, :2], features], axis=1)
    output.columns = ["label", "hand"] + list(features.columns)
    output["label"] = output["label"].str.strip()
    output["hand"] = output["hand"].str.strip()
    output.to_csv(output_path, index=False)
    print(f"Wrote {len(output)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply palm-size normalization to an old dataset.")
    parser.add_argument("input", help="CSV recorded with the old features")
    parser.add_argument("output", help="where to write the converted CSV")
    args = parser.parse_args()
    normalize(args.input, args.output)

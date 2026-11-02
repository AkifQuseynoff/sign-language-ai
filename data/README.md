# ASL Gesture Datasets

## Overview

This folder stores CSV datasets used to train the gesture classifiers. Each row contains a gesture label, detected hand side, and 60 geometric hand-landmark features.

The recording screen in the application lists the CSV files in this folder and appends new samples to the selected file.

## Dataset Files

### gestures_v2.csv

Dataset for three static hand gestures:

- Open hand
- Fist
- Pointing hand

### letters_v2.csv

Version 2 dataset for static ASL letters, including:

- A, B, C, D, F
- L, O, X, Y

## Recording Samples

In the application, choose a CSV file, enter a gesture name, and press **Start**. A sample is appended every seventh frame where a hand is detected.

New files should use the same CSV layout: `label`, `hand`, followed by the feature columns.

## Version 2 Features

In version 2, bone lengths and distances (`feature_10` to `feature_44`) are divided by the palm size, the distance from the wrist to the middle finger knuckle. This keeps the features the same whether the hand is near the camera or far from it. Because of this, `feature_18` (the palm itself) is always 1.

The v2 files were converted from the v1 recordings with `training/normalize_dataset.py`. Use the same script on any older CSV before training with it or recording into it.

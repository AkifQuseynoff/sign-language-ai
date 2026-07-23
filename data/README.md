# ASL Gesture Datasets

## Overview

This folder stores CSV datasets used to train the gesture classifiers. Each row contains a gesture label, detected hand side, and 60 geometric hand-landmark features.

The recording screen in the application lists the CSV files in this folder and appends new samples to the selected file.

## Dataset Files

### gestures.csv

Dataset for three static hand gestures:

- Open hand
- Fist
- Pointing hand

### letters_v1.csv

Version 1 dataset for static ASL letters, including:

- A, B, C, D, F
- L, O, X, Y

## Recording Samples

In the application, choose a CSV file, enter a gesture name, and press **Start**. A sample is appended every seventh frame where a hand is detected.

New files should use the same CSV layout: `label`, `hand`, followed by the feature columns.

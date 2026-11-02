# ASL Gesture Recognition Models

## Overview

This folder contains the MediaPipe hand-landmark model and trained gesture classifiers. The recognition screen lists the `.pkl` files here and reloads the selected classifier automatically.

## Model Files

### hand_landmarker.task

MediaPipe task model used to detect hands and return 21 landmarks per hand.

### gesture_model_v2.pkl

Version 2 classifier for three static gestures:

- Open hand
- Fist
- Pointing hand

### letters_model_v2.pkl

Version 2 classifier for static ASL letters, including A, B, C, D, F, L, O, X, and Y.

## Using a Model

Place a compatible scikit-learn `.pkl` model in this folder. It will appear in the recognition model selector the next time the application opens.

# ASL Gestures Translator

## Overview

This project is a real-time ASL gesture recognition system. It uses **MediaPipe** to detect hand landmarks and extracts geometric features such as joint angles, bone lengths, distances between fingertips, and finger direction vectors.

These features are used to create gesture datasets and train machine learning classifiers capable of recognizing static ASL gestures in real time.

## Motivation

This project explores how computer vision and machine learning can be used to improve accessibility by creating a system capable of recognizing ASL gestures and translating them into digital predictions in real time.

## Architecture

The system follows this pipeline:

```text
Camera
  ↓
MediaPipe Hand Landmarker
  ↓
21 Hand Landmarks
  ↓
Feature Extraction (60 features)
  ↓
Random Forest Classifier
  ↓
Gesture Prediction
```

## Project Structure

```text
ASL-gestures-translator/
│
├── main.py
├── record.py
│
├── camera/
│   └── camera.py
│
├── detection/
│   └── hand_detector.py
│
├── recognition/
│   ├── classifier.py
│   └── gesture_recognizer.py
│
├── features/
│   ├── feature_extractor.py
│   └── hand_geometry.py
│
├── data_collection/
│   └── data_collector.py
│
├── training/
│   └── train_model.py
│
├── dataset/
│
└── models/
```

## Project Components

### main.py

Uses trained models to recognize gestures from the camera feed in real time.

### camera/camera.py

Handles capturing frames from the camera.

### detection/hand_detector.py

Detects hands using MediaPipe and extracts hand landmarks.

### recognition/

- `classifier.py` uses the trained model to classify gestures.
- `gesture_recognizer.py` connects hand detection, feature extraction, and gesture classification to be used in `main.py`.

### record.py

Captures hand landmarks from the camera, extracts features, and records samples into CSV datasets.

### training/train_model.py

Trains machine learning models using collected datasets and saves trained classifiers.

### features/hand_geometry.py

Contains mathematical functions for calculating distances, vectors, and angles between hand landmarks.

### features/feature_extractor.py

Uses hand geometry calculations to generate feature vectors from hand landmarks.

### data_collection/data_collector.py

Handles saving collected samples into dataset files.

### dataset/

Stores collected gesture datasets.

### models/

Stores trained machine learning models and MediaPipe model files.

## Feature Extraction

The system extracts **60 geometric features** from each detected hand:

- 10 joint angles
- 20 bone lengths
- 10 fingertip-to-fingertip distances
- 5 wrist-to-fingertip distances
- 15 finger direction vector values

## Results

Current static letter classifier:

- Model: Random Forest Classifier
- Gestures: 10 ASL letters
- Samples: ~100 per letter
- Accuracy: 98%

## Requirements

The project requires:

- MediaPipe `hand_landmarker.task` model file
- Trained classifier (`.pkl`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Run the real-time gesture recognition system:

```bash
python main.py
```

To collect new training samples:

```bash
python record.py
```

To train the model:

```bash
python training/train_model.py
```
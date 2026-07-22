# **ASL gestures translator**


## Overview

This project is a real-time ASL gesture recognition system. It uses **MediaPipe** to detect hand landmarks and extracts geometric features such as joint angles, bone lengths, distances between fingertips, and finger direction vectors.

These features are used to create gesture datasets and train machine learning classifiers capable of recognizing static ASL gestures in real time.

## Architecture

Camera ->

MediaPipe Hand Landmarker ->

21 Hand Landmarks ->

Feature Extraction (60 features) ->

Random Forest Classifier ->

Gesture Prediction 

## Project Structure

### main.py
Uses trained models to recognize gestures from the camera feed in real time.

### record.py
Captures hand landmarks from the camera, extracts features, and records samples into CSV datasets.

### train_model.py
Trains machine learning models using collected datasets and saves trained classifiers.

### hand_geometry.py
Contains mathematical functions for calculating distances, vectors, and angles between hand landmarks.

### feature_extractor.py
Uses hand geometry calculations to generate feature vectors from hand landmarks.

### data_collector.py
Handles saving collected samples into dataset files.

### dataset/
Stores collected gesture datasets.

### models/
Stores trained machine learning models and MediaPipe model files.

## Feature Extraction

The system extracts 60 geometric features from each detected hand:

- 10 joint angles
- 20 bone lengths
- 10 fingertip-to-fingertip distances
- 5 wrist-to-fingertip distances
- 15 normalized finger direction vectors

## Results

Current static letter classifier:

- Model: Random Forest Classifier
- Gestures: 10 ASL letters
- Samples: 100 per letter
- Accuracy: 98%

## Requirements

The project requires:

- MediaPipe `hand_landmarker.task` model file
- Trained classifier (`.pkl`)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python main.py
```
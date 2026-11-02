# ASL Translator

## Overview

ASL Translator is a desktop application for recognizing static ASL gestures and collecting new gesture samples. It uses MediaPipe hand landmarks, geometric feature extraction, and trained scikit-learn classifiers.

The application includes live camera preview, prediction confidence, model selection, and a recording workspace for extending the datasets.

## Architecture

```text
Camera
  ↓
MediaPipe Hand Landmarker
  ↓
21 Hand Landmarks
  ↓
Feature Extraction (60 features)
  ↓
Selected Classifier Model
  ↓
Prediction and Confidence
```

## Project Structure

```text
sign-language-ai/
├── UI/                 # Tkinter desktop application
├── camera/             # OpenCV camera access
├── data/               # Gesture CSV datasets
├── data_collection/    # CSV sample writer
├── detection/          # MediaPipe hand detection
├── feature_extraction/ # Landmark feature generation
├── models/             # Classifier and MediaPipe model files
├── recognition/        # Gesture classification pipeline
├── training/           # Model training script
├── main.py             # Reusable recognition pipeline and preview
└── record.py           # Original command-line recording script
```

## Application Features

- Live recognition with prediction and confidence values.
- Model selector for `.pkl` files in `models/`.
- Recording workspace with camera preview, gesture name, dataset selector, and sample counter.
- Samples are saved to the chosen CSV every seventh frame where a hand is detected.

## Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

The project also requires `models/hand_landmarker.task` and at least one trained `.pkl` classifier.

## Running the Application

Start the desktop application:

```bash
python -m UI.app
```

You may need to allow camera access when prompted by your operating system.

## Other Commands

Run the OpenCV recognition preview:

```bash
python main.py
```

Train a model from a dataset:

```bash
python training/train_model.py --data data/letters_v2.csv --output models/letters_model_v2.pkl
```

The script tests on the last 20% of each gesture's samples, not a random 20%, because neighbouring samples are nearly identical and a random split overstates accuracy. After testing, it trains on all samples and saves the model.

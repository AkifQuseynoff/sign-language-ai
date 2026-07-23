# ASL Translator User Interface

## Overview

The UI is a Tkinter desktop application started with:

```bash
python -m UI.app
```

It provides a home screen, a live recognition workspace, and a gesture recording workspace.

## Views

### Home View

The home view provides navigation to recognition and recording.

### Recognition View

The recognition view displays the live camera preview, current prediction, and model confidence. Select a `.pkl` file from `models/` to change the classifier; the recognizer restarts with the selected model.

### Recording View

The recording view displays the camera preview and lets you:

- Choose a CSV file from `data/`.
- Enter a gesture name.
- Start and stop sample collection.
- Track samples saved during the current session.

Samples are written through `data_collection/DataCollector` using the same feature extraction flow as `record.py`.

## Structure

- `app.py` starts the application.
- `main_window.py` manages navigation and camera cleanup.
- `views/home_view.py` contains the home screen.
- `views/recognition_view.py` contains live recognition.
- `views/recording_view.py` contains dataset recording.

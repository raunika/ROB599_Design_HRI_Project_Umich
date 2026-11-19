# DeepFace Edge Face Recognition & Tracking

A lightweight, real-time facial recognition, emotion detection, and head-tracking pipeline built with OpenCV and DeepFace. Designed for edge devices (such as Raspberry Pi or macOS/Linux hosts), the system identifies enrolled users, tracks their normalized frame position (`yaw` and `pitch`), detects positive facial expressions, and streams telemetry over local UDP sockets.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Enrolling Faces](#1-enrolling-faces)
  - [2. Running Real-Time Recognition](#2-running-real-time-recognition)
- [Configuration](#configuration)
- [Telemetry & Output Formats](#telemetry--output-formats)

---

## Overview

The project consists of two scripts:

1. **Enrollment Script (`enroll.py`):** Processes single photos or batches of images, isolates the largest detected face in each image, and stores cropped face samples in a structured local database directory.
2. **Recognition & Tracking Script (`tracker.py`):** Captures live video asynchronously, matches detected faces against the enrolled database using `SFace`, evaluates positive emotion stability over a rolling window, and broadcasts tracking coordinates and presence data via UDP.

---

## Project Structure

```text
.
├── enroll.py               # Face cropping and database enrollment CLI tool
├── tracker.py              # Real-time face recognition, tracking, and UDP streamer
├── README.md
└── database/               # Auto-created local face database
    └── <Person_Name>/      # Subfolder per enrolled individual
        └── <Person_Name>_<timestamp>.jpg
```

---

## Installation

1. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note:** On the first run, `DeepFace` will automatically download the `SFace` recognition and emotion model weights.

---

## Usage

### 1. Enrolling Faces

Before running live recognition, populate the `./database` directory with cropped face images using the enrollment script.

**Enroll from a single image:**
```bash
python enroll.py --input path/to/photo.jpg --name Alice
```

**Batch enroll from a directory of images:**
```bash
python enroll.py --input path/to/alice_photos/ --name Alice
```

**CLI Arguments:**

| Argument | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `--input` | Yes | — | Path to a single image (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) or a folder of images. |
| `--name` | Yes | — | Name of the person (creates `./database/<name>/`). |
| `--db` | No | `./database` | Root directory for the face database. |

---

### 2. Running Real-Time Recognition

Start the live camera tracker and UDP broadcaster:
```bash
python tracker.py
```

**Run without perception UDP broadcasting (stdout & presence port only):**
```bash
python tracker.py --no-socket
```

Press **`q`** while the OpenCV preview window is focused to cleanly release the camera and exit.

---

## Configuration

Edit the constants at the top of `tracker.py` to tune hardware, model, and tracking behavior:

| Constant | Default | Description |
| :--- | :--- | :--- |
| `DB_PATH` | `"./database"` | Path to the enrolled face database directory. |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `640` / `480` | Camera capture resolution. |
| `TARGET_FPS` | `10` | Target main loop processing rate. |
| `OUTPUT_HZ` | `2.0` | Frequency (in Hz) for JSON stdout logs and UDP packets. |
| `PREFERRED_CAMERA_INDEX` | `1` | Preferred webcam index (automatically falls back to `0–2` if unavailable). |
| `PRIMARY_USER` | `"User"` | Target subfolder name to lock onto. Set to `""` or `None` to track any detected face. |
| `MODEL_NAME` | `'SFace'` | DeepFace recognition embedding model. |
| `DETECTOR_BACKEND` | `'opencv'` | Face detection backend used for extraction and matching. |
| `DEADZONE_WIDTH` | `0.5` | Normalized center region (`0.0–1.0`) where `yaw` and `pitch` remain locked at `0.5`. |
| `SMOOTHING_ALPHA` | `0.3` | EMA smoothing factor (`0.0–1.0`) for bounding boxes and coordinates. |
| `RECOGNITION_INTERVAL` | `0.5` | Minimum seconds between database identity lookups. |
| `EMOTION_INTERVAL` | `0.5` | Minimum seconds between facial emotion analyses. |
| `PRESENCE_PORT` | `5006` | Local UDP port for presence heartbeat packets. |
| `PRESENCE_HOLD_SECONDS` | `1.5` | Grace period (in seconds) before marking a user absent after losing sight of them. |

---

## Telemetry & Output Formats

At `OUTPUT_HZ` (`2.0 Hz` by default), the recognition script emits data to three local channels:

### 1. Standard Output (`stdout`)
Prints a JSON line to the console:
```json
{
  "timestamp": 1731982000.45,
  "name": "User",
  "emotion": true,
  "yaw": 0.5,
  "pitch": 0.5312
}
```

### 2. Perception Socket (`127.0.0.1:5005` via UDP)
Active unless `--no-socket` is passed. Intended for servo/gimbal or avatar gaze control (`0.5` is dead center):
```json
{
  "yaw": 0.5,
  "pitch": 0.5312,
  "user_name": "User",
  "in_frame": true,
  "is_happy": true
}
```

### 3. Presence Socket (`127.0.0.1:5006` via UDP)
Provides debounced conversational presence state for downstream LLM/assistant processes:
```json
{
  "name": "User",
  "present": true,
  "positive": true,
  "timestamp": 1731982000.45
}
```

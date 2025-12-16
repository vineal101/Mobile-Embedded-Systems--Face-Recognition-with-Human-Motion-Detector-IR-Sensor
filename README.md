## Mobile Embedded Systems: Face Recognition with Human Motion Detector

This project implements a resource-optimized face authentication system for the Raspberry Pi 4, utilizing a Passive Infrared (PIR) sensor to trigger the face recognition pipeline.

## 1. System Overview

The system is designed to minimize CPU and power consumption by only activating the camera and the computationally intensive face recognition models when the PIR sensor detects human motion.

Key components:
* **Face Detection:** YuNet ONNX model (`face_detection_yunet_2023mar.onnx`)
* **Face Embedding:** ArcFace ONNX model (`arcface_r100.onnx`)
* **User Data Storage:** Normalized feature vectors (`.npy` files)
* **Optimization:** Specific camera settings (640x480, 40 FPS, CAP_V4L2) and motion gating for RPi 4.

## 2. Code Modules

### 2.1. `yuCapture.py` (User Enrollment - `enroll.py`)

This script handles the process of enrolling a new user by generating a highly representative face embedding.

| Configuration | Value | Purpose |
| :--- | :--- | :--- |
| `TOTAL_SAMPLES` | 40 | Number of face samples to collect. |
| `MIN_FACE_SIZE` | 90 | Minimum required face size in pixels. |
| `BLUR_THRESHOLD` | 70.0 | Minimum Laplacian variance for a non-blurry image. |

**Process:**
1.  Prompts for a `USERNAME` and creates the directory `data/users/{USERNAME}`.
2.  Continuously captures frames, performs face detection (YuNet), and applies quality checks (size and blur).
3.  If quality checks pass, the face is processed, and the ArcFace model generates an L2-normalized feature vector.
4.  After collecting 40 samples, the script calculates a single **mean normalized embedding** from all vectors.
5.  This mean vector is saved as `embedding.npy` in the user's directory for authentication.

### 2.2. `yuAuth.py` (Face Authentication - `recognize.py`)

This is the primary script for real-time face verification against the enrolled user database.

| Configuration | Value | Purpose |
| :--- | :--- | :--- |
| `THRESH` | 0.6 | Cosine similarity score required for a match. |
| `VOTE_WINDOW` | 6 | Number of recent frames used for temporal voting. |
| `VOTE_PASS` | 4 | Number of positive votes required in the window for authentication. |

**Process:**
1.  All `embedding.npy` files from `data/users` are loaded into memory.
2.  The script captures video, detects a face (YuNet), and generates a real-time embedding (ArcFace).
3.  The embedding is compared to all loaded user embeddings using **Cosine Similarity**.
4.  **Temporal Voting:** The match result (`similarity > THRESH`) is fed into a queue of size 6. Authentication is successful only if the count of positive matches reaches 4, ensuring stability and reducing false positives.

### 2.3. `pir.py` (PIR Sensor Test)

A utility to verify the functionality of the Passive Infrared (PIR) motion sensor connected to the Raspberry Pi's GPIO pins.

**Process:**
1.  Sets GPIO mode to BCM and configures `PIR_PIN` (GPIO 25) as an input.
2.  Loops to read the pin state, printing "Motion Detected!" or "No Motion" every 0.5 seconds.

### 2.4. `sensorRecognize` (Complete Integration Script)

*(This file was not provided, but its function is described based on system requirements.)*

This script combines the motion detection and face authentication logic. Its primary goal is **resource optimization**. The face recognition camera and models remain inactive until motion is detected, significantly reducing CPU load and heat generation on the Raspberry Pi 4.

**Expected Flow:**
1.  Start by monitoring the PIR sensor (or a similar sensor like the one in `ultra.py`).
2.  **IF Motion Detected:**
    * Activate the camera.
    * Begin the face authentication loop from `yuAuth.py`.
3.  **IF Authentication Success:**
    * Trigger an event (e.g., log access, open a relay).
4.  **IF No Motion OR Timeout:**
    * Deactivate the camera/recognition loop to return to a low-power state.

## 3. Models and Data

* **Models Directory (`models/`):** Contains the pre-trained neural network models for computer vision tasks.
    * `face_detection_yunet_2023mar.onnx` (YuNet)
    * `arcface_r100.onnx` (ArcFace)
* **Data Directory (`data/users/`):** Stores all enrolled user data.
    * `data/users/{USERNAME}/embedding.npy`: A single 512-dimensional, L2-normalized feature vector representing the mean face of the enrolled user.
* **Other Sensor Scripts:** `ultra.py` is mentioned for testing other sensors (likely an ultrasonic or proximity sensor) that can also be used to trigger the face recognition pipeline. Mobile-Embedded-Systems--Face-Recognition-with-Human-Motion-Detector-IR-Sensor

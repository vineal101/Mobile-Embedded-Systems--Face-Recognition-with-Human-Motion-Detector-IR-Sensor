# Mobile Embedded Systems: Face Recognition with Human Motion Detector

This project implements a resource-optimized face authentication system for the Raspberry Pi 4. It uses a multi-sensor array (PIR + Ultrasonic) to gate the power-intensive face recognition pipeline, ensuring the system runs efficiently.

## 1. System Overview

The system is designed to minimize CPU and power consumption by only activating the camera and deep learning models when a user is physically close and moving.

Key components:
* **Face Detection:** YuNet ONNX model (`face_detection_yunet_2023mar.onnx`) running at reduced resolution (320x240).
* **Face Embedding:** MobileFaceNet (`w600k_mbf.onnx`) from the [InsightFace buffalo_sc pack](https://github.com/deepinsight/insightface/tree/master/model_zoo). This model is highly optimized for ARM CPUs.
* **User Data Storage:** A matrix of normalized feature vectors (`.npy` files) representing various angles of the user's face.
* **Sensors:** HC-SR04 Ultrasonic Sensor (Distance) and HC-SR501 PIR Sensor (Motion).

## 2. Code Modules

### 2.1. `enroll.py` (User Enrollment)

Captures a robust dataset of the user's face to create a biometric template.

| Configuration | Value | Purpose |
| :--- | :--- | :--- |
| `TOTAL_SAMPLES` | 40 | Total embeddings to collect per user. |
| `STARE_DURATION` | 4.0s | Time dedicated to frontal face capture. |
| `BLUR_THRESHOLD` | 30.0 | Strict laplacian variance check to reject blurry frames. |

**Process:**
1.  **Phase 1 (Stare):** Captures 15 samples of the user looking straight ahead.
2.  **Phase 2 (Rotate):** The user rotates their head. The script performs a **Diversity Check** (`sims.max() > 0.80`), rejecting frames that are too similar to existing ones. This ensures the template covers the full 3D profile of the face.
3.  **Storage:** Saves a stack of 40 embeddings (shape `40x512`) into `embeddings.npy`.

### 2.2. `recognize.py` (Face Authentication)

The core recognition script optimized for high FPS on the Pi 4.

| Configuration | Value | Purpose |
| :--- | :--- | :--- |
| `THRESH` | 0.65 | Similarity threshold for a match. |
| `SKIP_FRAMES` | 2 | Process 1 frame, skip 2 to save CPU. |
| `DETECT_W/H` | 320x240 | Detection runs at low res for speed; alignment maps back to high res. |

**Process:**
1.  Loads `embeddings.npy` for all users into memory.
2.  Captures video and detects faces using YuNet.
3.  Aligns the face and extracts a 512-D vector using MobileFaceNet.
4.  **Matrix Comparison:** Calculates the dot product between the live vector and *all* stored user vectors simultaneously (`templates @ vec`).
5.  **Voting:** Uses a sliding window (size 10) to confirm identity. Requires 6 positive matches to authenticate.

### 2.3. `sensorRecognize.py` (Complete Integration)

The master script that integrates sensors with the recognition logic.

**Workflow:**
1.  **Sensor Polling:** Continuously reads the Ultrasonic and PIR sensors.
2.  **Trigger Condition:** `person_present` is TRUE if:
    * Distance is < 70cm (Ultrasonic)
    * Motion is detected (PIR)
3.  **Activation:** If triggered, the camera thread starts, and models are loaded (or utilized if pre-loaded).
4.  **Timeout:** If the user leaves (sensors go low), an 8-second timer counts down. If no one returns, the camera is released to save power.

### 2.4. Utility Scripts

* **`ultra.py`**: A diagnostic tool that outputs raw data from the PIR and Ultrasonic sensors to help calibrate the distance threshold.
* **`pir.py`**: A simple test script to verify GPIO connectivity for the PIR sensor.

## 3. Models and Data

* **Models Directory (`models/`):**
    * `face_detection_yunet_2023mar.onnx`: Lightweight face detector.
    * `w600k_mbf.onnx`: MobileFaceNet (from `buffalo_sc`), optimized for edge devices.
* **Data Directory (`data/users/`):**
    * `{USERNAME}/embeddings.npy`: Contains the raw stack of feature vectors (not a mean vector), allowing for accurate multi-angle matching.

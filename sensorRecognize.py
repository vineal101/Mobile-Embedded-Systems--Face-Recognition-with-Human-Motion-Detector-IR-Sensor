#!/usr/bin/env python3
"""
Integrated Face Authentication System
- Ultrasonic + PIR sensors trigger camera activation
- Camera turns ON when person within 70cm AND motion detected
- Camera turns OFF after 8 seconds of no presence
"""
import time
import threading
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import RPi.GPIO as GPIO

try:
    import onnxruntime as ort
    USE_ORT = True
except ImportError:
    USE_ORT = False

# ================= SENSOR CONFIG =================
TRIG = 23
ECHO = 24
PIR_PIN = 25

DISTANCE_THRESHOLD_CM = 70.0
SENSOR_SAMPLE_INTERVAL = 0.1
SENSOR_LOG_INTERVAL = 2.0
IDLE_TIMEOUT = 8.0

# ================= FACE AUTH CONFIG =================
USERS_DIR = Path("data/users")
THRESH = 0.65
VOTE_WINDOW = 10
VOTE_PASS = 6

CAM_W, CAM_H = 640, 480
DETECT_W, DETECT_H = 320, 240
SKIP_FRAMES = 2

YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"
EMBED_MODEL = "models/w600k_mbf.onnx"

ARCFACE_REF = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# ================= SENSOR MANAGER =================
class SensorManager:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        
        # Ultrasonic
        GPIO.setup(TRIG, GPIO.OUT)
        GPIO.setup(ECHO, GPIO.IN)
        GPIO.output(TRIG, False)
        
        # PIR - simple setup, no pull up/down
        GPIO.setup(PIR_PIN, GPIO.IN)
    
    def get_distance(self) -> float:
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        
        pulse_start = time.time()
        pulse_end = time.time()
        timeout = time.time() + 0.1
        
        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()
            if pulse_start > timeout:
                return 999.0
        
        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()
            if pulse_end > timeout:
                return 999.0
        
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        return round(distance, 2)
    
    def get_motion(self) -> bool:
        # Simple check - HIGH means motion detected
        return GPIO.input(PIR_PIN) == 1
    
    def read_all(self) -> tuple:
        distance = self.get_distance()
        motion = self.get_motion()
        in_range = distance <= DISTANCE_THRESHOLD_CM
        person_present = in_range and motion
        return distance, motion, in_range, person_present
    
    def cleanup(self):
        GPIO.cleanup()


# ================= FACE AUTH =================
def l2_normalize(v, eps=1e-10):
    norm = np.linalg.norm(v)
    return v / (norm + eps)


def align_face_112(frame, lm5):
    M, _ = cv2.estimateAffinePartial2D(lm5, ARCFACE_REF, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR)


class Embedder:
    def __init__(self, model_path):
        if USE_ORT:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.use_ort = True
        else:
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.use_ort = False
    
    def embed(self, face):
        if self.use_ort:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            blob = (face_rgb.astype(np.float32) - 127.5) / 128.0
            blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
            out = self.session.run(None, {self.input_name: blob})[0]
        else:
            blob = cv2.dnn.blobFromImage(face, 1/128.0, (112,112), (127.5,127.5,127.5), swapRB=True)
            self.net.setInput(blob)
            out = self.net.forward()
        return l2_normalize(out.flatten())


class CameraManager:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.frame = None
        self.frame_count = 0
        self.lock = threading.Lock()
        self._thread = None
    
    def start(self):
        if self.is_running:
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        time.sleep(0.5)
    
    def _capture_loop(self):
        while self.is_running:
            if self.cap is not None:
                ok, f = self.cap.read()
                if ok:
                    with self.lock:
                        self.frame = f
                        self.frame_count += 1
    
    def read(self):
        with self.lock:
            if self.frame is None:
                return None, 0
            return self.frame.copy(), self.frame_count
    
    def stop(self):
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.frame = None
        self.frame_count = 0


def load_users():
    users = {}
    if not USERS_DIR.exists():
        return users
    for d in USERS_DIR.iterdir():
        if d.is_dir():
            p = d / "embeddings.npy"
            if p.exists():
                users[d.name] = np.load(p).astype(np.float32)
    return users


def find_embedding_model():
    for p in [EMBED_MODEL, "models/webface_r50.onnx", "models/buffalo_sc/w600k_mbf.onnx"]:
        if Path(p).exists():
            return p
    return None


# ================= MAIN SYSTEM =================
def main():
    print()
    print("Face Authentication System")
    print("-" * 40)
    print()
    
    # Initialize
    print("Starting up...")
    
    print("  Setting up sensors...")
    sensors = SensorManager()
    print("  Waiting for PIR to stabilize...")
    time.sleep(2)
    print("  Sensors ready")
    
    print("  Loading models...")
    embed_path = find_embedding_model()
    if embed_path is None:
        print("  Error: No embedding model found")
        sensors.cleanup()
        return
    
    users = load_users()
    if not users:
        print("  Error: No users enrolled")
        sensors.cleanup()
        return
    
    print(f"  Loaded {len(users)} user(s)")
    
    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL, "", (DETECT_W, DETECT_H),
        score_threshold=0.6, nms_threshold=0.3, top_k=1
    )
    embedder = Embedder(embed_path)
    print("  Models ready")
    print()
    print("System ready. Waiting for person...")
    print()
    
    camera = CameraManager()
    votes = deque(maxlen=VOTE_WINDOW)
    
    # State
    camera_active = False
    no_presence_start = None
    last_processed = 0
    last_sensor_log = 0
    label = "UNKNOWN"
    score = 0.0
    last_bbox = None
    
    try:
        while True:
            now = time.time()
            
            # Read sensors
            distance, motion, in_range, person_present = sensors.read_all()
            
            # Log sensor status every 2 seconds when idle
            if not camera_active and (now - last_sensor_log) >= SENSOR_LOG_INTERVAL:
                last_sensor_log = now
                print(f"  Distance: {distance:6.1f}cm | In range: {str(in_range):5} | Motion: {str(motion):5} | Person: {str(person_present):5}")
            
            # IDLE - camera off, waiting for person
            if not camera_active:
                if person_present:
                    print()
                    print("Person detected - starting camera...")
                    
                    camera.start()
                    time.sleep(1.0)
                    
                    votes.clear()
                    label = "UNKNOWN"
                    score = 0.0
                    last_bbox = None
                    no_presence_start = None
                    camera_active = True
                    
                    print("Authenticating...")
                    print()
                else:
                    time.sleep(SENSOR_SAMPLE_INTERVAL)
                    continue
            
            # ACTIVE - camera on, authenticating
            if camera_active:
                # Check if person left
                if not person_present:
                    if no_presence_start is None:
                        no_presence_start = now
                        print(f"No person detected - waiting {IDLE_TIMEOUT}s before stopping camera...")
                    
                    elapsed = now - no_presence_start
                    if elapsed >= IDLE_TIMEOUT:
                        print("Timeout reached - stopping camera")
                        print()
                        
                        cv2.destroyAllWindows()
                        camera.stop()
                        
                        votes.clear()
                        label = "UNKNOWN"
                        score = 0.0
                        last_bbox = None
                        camera_active = False
                        last_sensor_log = 0
                        
                        print("Waiting for person...")
                        print()
                        continue
                else:
                    if no_presence_start is not None:
                        print("Person returned - continuing")
                    no_presence_start = None
                
                # Process camera frame
                frame, frame_num = camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Skip frames
                if (frame_num - last_processed) >= SKIP_FRAMES:
                    last_processed = frame_num
                    
                    # Face detection
                    small = cv2.resize(frame, (DETECT_W, DETECT_H))
                    detector.setInputSize((DETECT_W, DETECT_H))
                    _, faces = detector.detect(small)
                    
                    if faces is not None and len(faces) > 0:
                        f = faces[0]
                        
                        # Scale landmarks
                        lm = f[4:14].reshape(5, 2)
                        lm[:, 0] *= CAM_W / DETECT_W
                        lm[:, 1] *= CAM_H / DETECT_H
                        
                        # Store bbox
                        sx, sy = CAM_W / DETECT_W, CAM_H / DETECT_H
                        last_bbox = (
                            int(f[0] * sx), int(f[1] * sy),
                            int((f[0] + f[2]) * sx), int((f[1] + f[3]) * sy)
                        )
                        
                        # Get embedding
                        aligned = align_face_112(frame, lm)
                        if aligned is not None:
                            vec = embedder.embed(aligned)
                            
                            best_user, best_sim = None, -1
                            for name, templates in users.items():
                                sims = templates @ vec
                                s = sims.max()
                                if s > best_sim:
                                    best_sim = s
                                    best_user = name
                            
                            votes.append(best_sim >= THRESH)
                            if sum(votes) >= VOTE_PASS:
                                label = best_user
                                score = best_sim
                            else:
                                label = "UNKNOWN"
                                score = best_sim if best_sim > 0 else 0.0
                    else:
                        last_bbox = None
                
                # Draw on frame
                if last_bbox is not None:
                    x1, y1, x2, y2 = last_bbox
                    color = (0, 255, 0) if label != "UNKNOWN" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                color = (0, 255, 0) if label != "UNKNOWN" else (0, 0, 255)
                cv2.putText(frame, f"{label} {score:.2f}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                
                cv2.imshow("Face Auth", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        print()
        print("Shutting down...")
        cv2.destroyAllWindows()
        camera.stop()
        sensors.cleanup()
        print("Done")
        print()


if __name__ == "__main__":
    main()

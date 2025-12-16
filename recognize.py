#!/usr/bin/env python3
"""
Pi4-Optimized Face Recognition v2
=======================================
Strategy: Since embedding models (ResNet-50 vs MobileFaceNet) have similar 
inference times on ARM CPU, we optimize by:
1. Much lower detection resolution (160x120)
2. Aggressive frame skipping (process 1 in 4 frames)
3. Run embedding only when face is stable
4. Use grayscale for detection
5. ONNX Runtime optimizations
6. Optional: Use buffalo_sc (16MB) from InsightFace if available

Download buffalo_sc (has MobileFaceNet):
  wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip
  unzip buffalo_sc.zip -d models/buffalo_sc/
"""
import os
import time
import threading
from pathlib import Path
from collections import deque

import cv2
import numpy as np

# Try ONNX Runtime first (faster), fallback to cv2.dnn
try:
    import onnxruntime as ort
    USE_ORT = True
except ImportError:
    USE_ORT = False
    print("onnxruntime not found, using cv2.dnn (slower)")

# ================= CONFIG =================
USERS_DIR = Path("data/users")

THRESH = 0.65               # Stricter threshold (original value)
VOTE_WINDOW = 10            # Larger window for stability  
VOTE_PASS = 6               # Require 6/10 positive votes

# Camera settings (keep full res for display only)
CAM_W, CAM_H = 640, 480

# Detection resolution - 320x240 gives better landmark accuracy for alignment
# This is critical for recognition accuracy (landmarks affect face alignment)
DETECT_W, DETECT_H = 320, 240

# Frame processing - less aggressive skipping for better accuracy
SKIP_FRAMES = 2             # Process every 2nd frame (was 3)

# Your existing models (or buffalo_sc models)
YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"
# EMBED_MODEL = "models/webface_r50.onnx"  # Original ResNet-50
EMBED_MODEL = "models/w600k_mbf.onnx"     # Try buffalo_sc MobileFaceNet if available

# Fallback paths if using buffalo_sc
BUFFALO_SC_DIR = Path("models/buffalo_sc")
# ==========================================

ARCFACE_REF = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def l2_normalize(v, eps=1e-10):
    norm = np.linalg.norm(v)
    return v / (norm + eps)


def align_face_112(frame, lm5):
    M, _ = cv2.estimateAffinePartial2D(lm5, ARCFACE_REF, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR)


class ONNXEmbedder:
    """ONNX Runtime embedder with Pi4 optimizations."""
    
    def __init__(self, model_path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Disable memory pattern for lower memory usage
        opts.enable_mem_pattern = False
        opts.enable_cpu_mem_arena = False
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        
        # Get expected input shape
        shape = self.session.get_inputs()[0].shape
        self.input_size = (shape[3], shape[2]) if len(shape) == 4 else (112, 112)
    
    def embed(self, face_aligned):
        # Preprocessing must match original cv2.dnn exactly for consistency
        # Original: scalefactor=1/128.0, mean=(127.5,127.5,127.5), swapRB=True
        # This is equivalent to: (img - 127.5) / 128.0 with BGR->RGB swap
        face_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
        blob = (face_rgb.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        
        out = self.session.run(None, {self.input_name: blob})[0]
        return l2_normalize(out.flatten())


class CVDNNEmbedder:
    """cv2.dnn embedder (fallback)."""
    
    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def embed(self, face_aligned):
        blob = cv2.dnn.blobFromImage(
            face_aligned, 1/128.0, (112, 112),
            mean=(127.5, 127.5, 127.5), swapRB=True
        )
        self.net.setInput(blob)
        out = self.net.forward().flatten()
        return l2_normalize(out)


class FrameGrabber:
    """Low-latency threaded camera grabber."""
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # MJPEG is faster on Pi
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        self.lock = threading.Lock()
        self.frame = None
        self.frame_count = 0
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
                    self.frame_count += 1

    def read(self):
        with self.lock:
            return (None, 0) if self.frame is None else (self.frame.copy(), self.frame_count)

    def release(self):
        self.running = False
        self.cap.release()


def load_users():
    users = {}
    for d in USERS_DIR.iterdir():
        if d.is_dir():
            p = d / "embeddings.npy"
            if p.exists():
                users[d.name] = np.load(p).astype(np.float32)
                print(f"  Loaded {d.name}: {users[d.name].shape[0]} templates")
    return users


def find_embedding_model():
    """Find the best available embedding model."""
    candidates = [
        EMBED_MODEL,
        "models/w600k_mbf.onnx",           # MobileFaceNet from buffalo_sc
        "models/buffalo_sc/w600k_mbf.onnx",
        "models/webface_r50.onnx",          # Original ResNet-50
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def main():
    print("=== Pi4 Face Auth v2 ===")
    print(f"Detection: {DETECT_W}x{DETECT_H}")
    print(f"Skip frames: {SKIP_FRAMES}")
    
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    
    # Load users
    users = load_users()
    if not users:
        print("No users enrolled.")
        return
    print(f"Loaded {len(users)} users")
    
    # Find models
    embed_path = find_embedding_model()
    if embed_path is None:
        print("ERROR: No embedding model found!")
        print("Download buffalo_sc: wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip")
        return
    print(f"Using embedding model: {embed_path}")
    
    # Initialize detector
    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL, "", (DETECT_W, DETECT_H),
        score_threshold=0.6, nms_threshold=0.3, top_k=1
    )
    
    # Initialize embedder
    if USE_ORT:
        embedder = ONNXEmbedder(embed_path)
    else:
        embedder = CVDNNEmbedder(embed_path)
    
    grab = FrameGrabber()
    votes = deque(maxlen=VOTE_WINDOW)
    
    # State tracking (persists across skipped frames)
    label = "UNKNOWN"
    score = 0.0
    last_processed = 0
    last_bbox = None  # Persist bbox for drawing on skipped frames
    
    print("Running... Press Q to quit")
    
    while True:
        frame, frame_num = grab.read()
        if frame is None:
            continue
        
        # Skip frames for performance
        should_process = (frame_num - last_processed) >= SKIP_FRAMES
        
        if should_process:
            last_processed = frame_num
            
            # Detect at 320x240 for accurate landmarks
            small = cv2.resize(frame, (DETECT_W, DETECT_H))
            detector.setInputSize((DETECT_W, DETECT_H))
            _, faces = detector.detect(small)
            
            if faces is not None and len(faces) > 0:
                f = faces[0]
                
                # Scale landmarks to full resolution
                lm = f[4:14].reshape(5, 2)
                lm[:, 0] *= CAM_W / DETECT_W
                lm[:, 1] *= CAM_H / DETECT_H
                
                # Store bbox for persistent drawing
                sx, sy = CAM_W / DETECT_W, CAM_H / DETECT_H
                last_bbox = (
                    int(f[0] * sx), int(f[1] * sy),
                    int((f[0] + f[2]) * sx), int((f[1] + f[3]) * sy)
                )
                
                # Run embedding on every detection (like original)
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
                    
                    # Voting system for stability
                    votes.append(best_sim >= THRESH)
                    if sum(votes) >= VOTE_PASS:
                        label = best_user
                        score = best_sim
                    else:
                        label = "UNKNOWN"
                        score = best_sim if best_sim > 0 else 0.0
            else:
                last_bbox = None
        
        # Draw persistent bbox (even on skipped frames)
        if last_bbox is not None:
            x1, y1, x2, y2 = last_bbox
            color = (0, 255, 0) if label != "UNKNOWN" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw persistent label (even on skipped frames)
        color = (0, 255, 0) if label != "UNKNOWN" else (0, 0, 255)
        cv2.putText(frame, f"{label} {score:.2f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        cv2.imshow("Auth", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break
    
    grab.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

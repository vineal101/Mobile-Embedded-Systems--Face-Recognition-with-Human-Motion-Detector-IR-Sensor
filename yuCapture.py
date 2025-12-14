import cv2
import numpy as np
import os
import time

# ---------------- CONFIG ----------------
BASE_DIR = "data/users"
TOTAL_SAMPLES = 40
MIN_FACE_SIZE = 90
BLUR_THRESHOLD = 70.0
STARTUP_DELAY = 5
SAMPLE_INTERVAL = 0.12  # 24 FPS => ~every 3 frames

# ---------------------------------------
def is_blurry(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

def l2(v):
    return v / np.linalg.norm(v)

# ---------------------------------------
username = input("Enter username to enroll: ").strip()
user_dir = os.path.join(BASE_DIR, username)
os.makedirs(user_dir, exist_ok=True)

# ---------------- Face Detector & Embedder ----------------
detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (640, 480),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=1
)

embedder = cv2.dnn.readNetFromONNX("models/arcface_r100.onnx")

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 40)

print(f"Starting enrollment for '{username}'")
print("Please get ready and rotate your head naturally.")
time.sleep(STARTUP_DELAY)

mbeddings = []
last_sample_time = 0

while len(embeddings) < TOTAL_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is not None and len(faces) == 1:
        x, y, fw, fh = map(int, faces[0][:4])
        face = frame[y:y+fh, x:x+fw]

        if fw > MIN_FACE_SIZE and not is_blurry(face):
            now = time.time()
            if now - last_sample_time > SAMPLE_INTERVAL:
                # ArcFace expects 112x112
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(face, (112,112)),
                    1/255.0, (112,112), swapRB=True, crop=False
                )
                embedder.setInput(blob)
                vec = embedder.forward().flatten()
                embeddings.append(l2(vec))
                last_sample_time = now

        # Draw face rectangle
        cv2.rectangle(frame,(x,y),(x+fw,y+fh),(0,255,0),2)

    cv2.putText(frame, f"Samples {len(embeddings)}/{TOTAL_SAMPLES}", 
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Enroll User", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

# Save mean normalized embedding
mean_embedding = l2(np.mean(embeddings, axis=0))
np.save(os.path.join(user_dir, "embedding.npy"), mean_embedding)

print(f"Enrollment complete for '{username}'")


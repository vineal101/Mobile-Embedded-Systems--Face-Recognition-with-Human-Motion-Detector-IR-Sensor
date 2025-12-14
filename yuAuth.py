import cv2
import numpy as np
import os
from collections import deque

# ---------------- CONFIG ----------------
USERS_DIR = "data/users"
THRESH = 0.6
VOTE_WINDOW = 6
VOTE_PASS = 4

# ---------------------------------------
def l2(v):
    return v / np.linalg.norm(v)

def cosine(a, b):
    return np.dot(a, b)

# ---------------------------------------
# Load all users
users = {}
for name in os.listdir(USERS_DIR):
    emb_path = os.path.join(USERS_DIR, name, "embedding.npy")
    if os.path.exists(emb_path):
        users[name] = np.load(emb_path)

print("Loaded users:", list(users.keys()))

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

votes = deque(maxlen=VOTE_WINDOW)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    label = "UNKNOWN"
    score = 0.0

    if faces is not None and len(faces) == 1:
        x, y, fw, fh = map(int, faces[0][:4])
        face = frame[y:y+fh, x:x+fw]

        # ArcFace embedding
        blob = cv2.dnn.blobFromImage(
            cv2.resize(face,(112,112)),
            1/255.0, (112,112), swapRB=True
        )
        embedder.setInput(blob)
        vec = l2(embedder.forward().flatten())

        # Compare with all users
        best_user, best_sim = None, -1
        for name, emb in users.items():
            sim = cosine(vec, emb)
            if sim > best_sim:
                best_sim = sim
                best_user = name

        # Temporal voting
        votes.append(best_sim > THRESH)
        if sum(votes) >= VOTE_PASS:
            label = best_user
            score = best_sim

        cv2.rectangle(frame,(x,y),(x+fw,y+fh),(0,255,0),2)

    cv2.putText(frame, f"{label} {score:.2f}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0,255,0) if label!="UNKNOWN" else (0,0,255), 2)

    cv2.imshow("Face Authentication", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


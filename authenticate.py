import cv2
import numpy as np

cv2.setUseOptimized(True)
cv2.setNumThreads(2)

# Load models
detector = cv2.dnn.readNetFromCaffe(
    "face_detector.prototxt",
    "face_detector.caffemodel"
)
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

authorized = np.load("authorized_embedding.npy")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

THRESH = 0.6
DETECT_EVERY = 8  # run face detection every N frames

frame_count = 0
faces_cache = []  # [(x1,y1,x2,y2, sim)]

print("Running optimized face authentication... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # ---------------------------
    # Face detection (throttled)
    # ---------------------------
    if frame_count % DETECT_EVERY == 0:
        faces_cache.clear()

        small = cv2.resize(frame, (320, 240))
        blob = cv2.dnn.blobFromImage(
            small, 1.0, (300, 300), (104, 177, 123)
        )
        detector.setInput(blob)
        detections = detector.forward()

        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf < 0.7:
                continue

            box = detections[0, 0, i, 3:7] * np.array([320,240,320,240])
            x1, y1, x2, y2 = box.astype(int)

            # Scale back to original frame
            x1 = int(x1 * w / 320)
            x2 = int(x2 * w / 320)
            y1 = int(y1 * h / 240)
            y2 = int(y2 * h / 240)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(
                cv2.resize(face, (96,96)),
                1.0/255, (96,96),
                (0,0,0),
                swapRB=True,
                crop=False
            )

            embedder.setInput(face_blob)
            vec = embedder.forward()

            sim = cosine_similarity(authorized.flatten(), vec.flatten())
            faces_cache.append((x1,y1,x2,y2,sim))

    # ---------------------------
    # Draw cached results
    # ---------------------------
    for (x1,y1,x2,y2,sim) in faces_cache:
        if sim > THRESH:
            color = (0,255,0)
            label = f"AUTHORIZED ({sim:.2f})"
        else:
            color = (0,0,255)
            label = f"UNKNOWN ({sim:.2f})"

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Face Authentication", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


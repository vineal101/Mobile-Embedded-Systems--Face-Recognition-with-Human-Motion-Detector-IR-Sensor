import cv2
import numpy as np

# Load face detector
detector = cv2.dnn.readNetFromCaffe(
    "face_detector.prototxt",
    "face_detector.caffemodel"
)

# Load OpenFace embedding model (NO prototxt needed)
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# Load the authorized 128-D vector
authorized = np.load("authorized_embedding.npy")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

THRESH = 0.6  # similarity threshold

print("Running face authentication... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300,300)),
        1.0, (300,300),
        (104,177,123)
    )
    detector.setInput(blob)
    detections = detector.forward()

    h, w = frame.shape[:2]

    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf < 0.7:
            continue

        x1, y1, x2, y2 = (detections[0,0,i,3:7] * np.array([w,h,w,h])).astype(int)

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

        if sim > THRESH:
            color = (0,255,0)  # green = authorized
            label = f"AUTHORIZED ({sim:.2f})"
        else:
            color = (0,0,255)  # red = unknown
            label = f"UNKNOWN ({sim:.2f})"

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Authentication", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


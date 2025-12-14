import cv2
import numpy as np
import time

# ------------------------
# Load face detector (SSD)
# ------------------------
detector = cv2.dnn.readNetFromCaffe(
    "face_detector.prototxt",
    "face_detector.caffemodel"
)

# ------------------------
# Load OpenFace embedding model
# ------------------------
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# ------------------------
# Open camera
# ------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

print("Press SPACE to capture your authorized face.")
print("Press ESC to exit.")

# ------------------------
# Main loop
# ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ------------------------
    # Detect faces
    # ------------------------
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0, (300, 300),
        (104, 177, 123)
    )
    detector.setInput(blob)
    detections = detector.forward()

    h, w = frame.shape[:2]
    face_box = None

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < 0.7:
            continue

        x1, y1, x2, y2 = (detections[0,0,i,3:7] * np.array([w,h,w,h])).astype(int)
        face_box = (x1, y1, x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Capture Authorized Face", frame)
    key = cv2.waitKey(10) & 0xFF

    # ------------------------
    # Spacebar = capture
    # ------------------------
    if key == 32:  # SPACEBAR
        if face_box is None:
            print("No face detected. Try again.")
            continue

        x1, y1, x2, y2 = face_box
        face = frame[y1:y2, x1:x2]

        # ------------------------
        # Create embedding
        # ------------------------
        face_blob = cv2.dnn.blobFromImage(
            cv2.resize(face, (96, 96)),
            1.0/255, (96, 96),
            (0,0,0),
            swapRB=True,
            crop=False
        )
        embedder.setInput(face_blob)
        vec = embedder.forward()

        np.save("authorized_embedding.npy", vec)
        print("Authorized face captured and saved to authorized_embedding.npy!")
        break

    # ESC = exit
    elif key == 27:
        print("Exiting without saving.")
        break

cap.release()
cv2.destroyAllWindows()


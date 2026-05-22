import cv2
import mediapipe as mp
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
mp_face_mesh = mp.solutions.face_mesh

NOSE = 1

face_ids = {}   # id → (x, y)
next_id = 0

def assign_id(x, y):
    global next_id

    # Se non ci sono ID, creane uno
    if not face_ids:
        face_ids[next_id] = (x, y)
        next_id += 1
        return next_id - 1

    # Cerca l'ID più vicino
    min_dist = float("inf")
    best_id = None

    for fid, (px, py) in face_ids.items():
        dist = np.hypot(x - px, y - py)
        if dist < min_dist:
            min_dist = dist
            best_id = fid

    # Se è vicino abbastanza → stesso volto
    if min_dist < 80:  # soglia in pixel
        face_ids[best_id] = (x, y)
        return best_id

    # Altrimenti è un volto nuovo
    face_ids[next_id] = (x, y)
    next_id += 1
    return next_id - 1


cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                lm = face_landmarks.landmark[NOSE]
                x = int(lm.x * w)
                y = int(lm.y * h)

                fid = assign_id(x, y)

                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID {fid}", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Face ID", frame)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()

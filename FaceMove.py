import cv2
import mediapipe as mp
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

mp_face_mesh = mp.solutions.face_mesh

NOSE = 1  # landmark del naso

# Dizionari per ID e storia
face_ids = {}        # id → (x, y)
face_history = {}    # id → lista di posizioni
next_id = 0          # ID da assegnare al prossimo volto

def assign_id(x, y):
    global next_id

    # Se non ci sono volti registrati → crea ID 0
    if not face_ids:
        face_ids[next_id] = (x, y)
        next_id += 1
        return next_id - 1

    # Trova l'ID più vicino
    min_dist = float("inf")
    best_id = None

    for fid, (px, py) in face_ids.items():
        dist = np.hypot(x - px, y - py)
        if dist < min_dist:
            min_dist = dist
            best_id = fid

    # Se è abbastanza vicino → stesso volto
    if min_dist < 80:  # soglia in pixel
        face_ids[best_id] = (x, y)
        return best_id

    # Altrimenti → nuovo volto
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

                # Estrai posizione del naso
                lm = face_landmarks.landmark[NOSE]
                x = int(lm.x * w)
                y = int(lm.y * h)

                # Assegna ID
                fid = assign_id(x, y)

                # Aggiorna storia
                if fid not in face_history:
                    face_history[fid] = []

                face_history[fid].append((x, y))

                # Limita la lunghezza della storia
                if len(face_history[fid]) > 50:
                    face_history[fid].pop(0)

                # Disegna traiettoria
                for i in range(1, len(face_history[fid])):
                    cv2.line(frame,
                             face_history[fid][i-1],
                             face_history[fid][i],
                             (0, 255, 255), 2)

                # Disegna ID e punto del naso
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID {fid}", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 0), 2)

        cv2.imshow("Face ID + Storia movimenti", frame)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()

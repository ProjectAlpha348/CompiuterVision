import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import json
import os
import warnings
import sys

warnings.filterwarnings("ignore", category=UserWarning)
print(sys.executable)

# ---------------------------
# 1) Carica MobileFaceNet
# ---------------------------
session = ort.InferenceSession("MobileFaceNet.onnx")
input_name = session.get_inputs()[0].name

def preprocess_face(face_img):
    face = cv2.resize(face_img, (112, 112))
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)
    return face

def get_embedding(face_img):
    face = preprocess_face(face_img)
    embedding = session.run(None, {input_name: face})[0]
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.flatten()

# ---------------------------
# 2) Crop migliorato
# ---------------------------
def safe_crop(frame, x1, y1, x2, y2, margin=0.25):
    h, w, _ = frame.shape

    bw = x2 - x1
    bh = y2 - y1

    x1 = int(max(0, x1 - bw * margin))
    y1 = int(max(0, y1 - bh * margin))
    x2 = int(min(w, x2 + bw * margin))
    y2 = int(min(h, y2 + bh * margin))

    return frame[y1:y2, x1:x2]

# ---------------------------
# 3) Carica embedding dal database
# ---------------------------
def load_embedding(name, db_path="database.json"):
    if not os.path.exists(db_path):
        print("Database non trovato.")
        return None

    with open(db_path, "r") as f:
        data = json.load(f)

    if name not in data:
        print(f"Embedding per '{name}' non trovato nel database.")
        return None

    emb = np.array(data[name], dtype=np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb

embedding_target = load_embedding("Umberto")

if embedding_target is None:
    print("Errore: impossibile caricare embedding dal database.")
    exit()

print("Embedding target caricato dal database!")

# ---------------------------
# 4) Realtime FaceID
# ---------------------------
mp_face = mp.solutions.face_detection
cap = cv2.VideoCapture(0)

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if results.detections:
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box

                x1 = int(box.xmin * w)
                y1 = int(box.ymin * h)
                x2 = int((box.xmin + box.width) * w)
                y2 = int((box.ymin + box.height) * h)

                face_crop = safe_crop(frame, x1, y1, x2, y2)

                if face_crop.size > 0:
                    embedding_live = get_embedding(face_crop)

                    dist = np.linalg.norm(embedding_target - embedding_live)
                    match = dist < 0.95

                    color = (0,255,0) if match else (0,0,255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    if match:
                        cv2.putText(frame, "Umberto", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    else:
                        cv2.putText(frame, "Sconosciuto", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.imshow("FaceID Realtime", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()

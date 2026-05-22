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
# 3) Database JSON
# ---------------------------
def load_all_embeddings(db_path="database.json"):
    if not os.path.exists(db_path):
        return {}
    with open(db_path, "r") as f:
        data = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}

def save_embedding(name, embedding, db_path="database.json"):
    data = {}
    if os.path.exists(db_path):
        with open(db_path, "r") as f:
            data = json.load(f)

    data[name] = embedding.tolist()

    with open(db_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Salvato nuovo ID nel database: {name}")

def create_new_id(db_embeddings):
    existing_ids = [k for k in db_embeddings.keys() if k.startswith("ID_")]
    if not existing_ids:
        return "ID_1"
    nums = [int(x.split("_")[1]) for x in existing_ids]
    return f"ID_{max(nums)+1}"

def find_best_match(embedding_live, db_embeddings, soglia=0.95):
    best_name = None
    best_dist = 999

    for name, emb in db_embeddings.items():
        dist = np.linalg.norm(embedding_live - emb)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    if best_dist < soglia:
        return best_name, best_dist
    else:
        return None, best_dist

# ---------------------------
# 4) Carica database
# ---------------------------
db_embeddings = load_all_embeddings()
print("Database caricato:", list(db_embeddings.keys()))

# ---------------------------
# 5) Realtime FaceID
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

                    # Cerca match nel database
                    name, dist = find_best_match(embedding_live, db_embeddings)

                    if name is not None:
                        # Persona riconosciuta
                        color = (0,255,0)
                        label = name
                    else:
                        # Nuova persona → crea ID
                        new_id = create_new_id(db_embeddings)
                        db_embeddings[new_id] = embedding_live
                        save_embedding(new_id, embedding_live)
                        color = (0,255,255)
                        label = new_id

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("FaceID Realtime", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()

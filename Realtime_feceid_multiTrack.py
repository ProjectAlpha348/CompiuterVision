import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import json
import os

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
# 5) Tracking ID
# ---------------------------
active_tracks = {}  # { "ID": {"bbox": (x1,y1,x2,y2), "missed": 0} }

def bbox_distance(b1, b2):
    x1,y1,x2,y2 = b1
    X1,Y1,X2,Y2 = b2
    cx1 = (x1+x2)/2
    cy1 = (y1+y2)/2
    cx2 = (X1+X2)/2
    cy2 = (Y1+Y2)/2
    return np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)

# ---------------------------
# 6) Realtime FaceID + Tracking
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

        seen_ids = []

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

                    # 1) Prova match embedding
                    name, dist = find_best_match(embedding_live, db_embeddings)

                    # 2) Se nessun match → nuovo ID
                    if name is None:
                        name = create_new_id(db_embeddings)
                        db_embeddings[name] = embedding_live
                        save_embedding(name, embedding_live)

                    # 3) Tracking: aggiorna posizione
                    active_tracks[name] = {"bbox": (x1,y1,x2,y2), "missed": 0}
                    seen_ids.append(name)

                    # 4) Disegna
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, name, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # 5) Incrementa missed per chi non è stato visto
        for ID in list(active_tracks.keys()):
            if ID not in seen_ids:
                active_tracks[ID]["missed"] += 1
                if active_tracks[ID]["missed"] > 15:
                    del active_tracks[ID]

        cv2.imshow("FaceID Realtime + Tracking", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()

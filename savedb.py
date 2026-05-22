import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import warnings
import json
import os
import sys

warnings.filterwarnings("ignore", category=UserWarning)
print(sys.executable)

# ---------------------------
# 1) Carica MobileFaceNet
# ---------------------------
session = ort.InferenceSession("MobileFaceNet.onnx")
input_name = session.get_inputs()[0].name

def save_embedding(name, embedding, db_path="database.json"):
    data = {}

    # Se il file esiste, lo carichiamo
    if os.path.exists(db_path):
        with open(db_path, "r") as f:
            data = json.load(f)

    # Convertiamo l'embedding in lista (JSON non supporta numpy)
    data[name] = embedding.tolist()

    # Salviamo
    with open(db_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Embedding salvato per: {name}")

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
    emb = emb / np.linalg.norm(emb)  # normalizzazione di sicurezza
    return emb


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
# 3) Genera embedding_target da volto.jpg
# ---------------------------
mp_face = mp.solutions.face_detection

img_ref = cv2.imread("volto.jpg")
if img_ref is None:
    print("Errore: impossibile caricare volto.jpg")
    exit()

h, w, _ = img_ref.shape

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if not results.detections:
        print("Nessun volto trovato in volto.jpg")
        exit()

    detection = results.detections[0]
    box = detection.location_data.relative_bounding_box

    x1 = int(box.xmin * w)
    y1 = int(box.ymin * h)
    x2 = int((box.xmin + box.width) * w)
    y2 = int((box.ymin + box.height) * h)

    face_crop_ref = safe_crop(img_ref, x1, y1, x2, y2)

    if face_crop_ref.size == 0:
        print("Errore nel crop del volto di riferimento")
        exit()

    embedding_target = get_embedding(face_crop_ref)
    print("Embedding target generato!")
    save_embedding("Umberto", embedding_target)


# ---------------------------
# 4) Realtime FaceID
# ---------------------------
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
                    match = dist < 0.95   # soglia leggermente più permissiva

                    color = (0,255,0) if match else (0,0,255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"dist: {dist:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("FaceID Realtime", frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()

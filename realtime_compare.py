'''
All’avvio dello script:

carichi volto.jpg

generi embedding di riferimento (embedding_target)

In realtime:

Mediapipe rileva il volto

MobileFaceNet genera embedding_live

confronti embedding_live con embedding_target

se distanza < soglia → verde

altrimenti → rosso
'''
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import warnings

# ---------------------------
# 1) Carica MobileFaceNet
# ---------------------------
session = ort.InferenceSession("MobileFaceNet.onnx")
input_name = session.get_inputs()[0].name
warnings.filterwarnings("ignore", category=UserWarning)
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

def same_person(emb1, emb2, soglia=0.9):
    dist = np.linalg.norm(emb1 - emb2)
    return dist < soglia, dist

# ---------------------------
# 2) Carica embedding da volto.jpg
# ---------------------------
img_ref = cv2.imread("volto.jpg")
embedding_target = get_embedding(img_ref)
print("Embedding di riferimento generato!")

# ---------------------------
# 3) Mediapipe Face Detection
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

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size > 0:
                    embedding_live = get_embedding(face_crop)
                    match, dist = same_person(embedding_target, embedding_live)

                    # Colore del rettangolo
                    color = (0,255,0) if match else (0,0,255)

                    # Disegna rettangolo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Mostra distanza
                    cv2.putText(frame, f"dist={dist:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("FaceID Realtime", frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()

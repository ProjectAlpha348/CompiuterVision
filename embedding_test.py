import cv2
import numpy as np
import onnxruntime as ort

# Carica il modello
session = ort.InferenceSession("MobileFaceNet.onnx")
input_name = session.get_inputs()[0].name

def preprocess_face(face_img):
    # Ridimensiona a 112x112
    face = cv2.resize(face_img, (112, 112))

    # Converti in float32
    face = face.astype(np.float32)

    # Normalizzazione [-1, 1]
    face = (face - 127.5) / 128.0

    # BGR → RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Cambia formato da HWC → CHW
    face = np.transpose(face, (2, 0, 1))

    # Aggiungi batch dimension
    face = np.expand_dims(face, axis=0)

    return face

def get_embedding(face_img):
    face = preprocess_face(face_img)

    # Inferenza ONNX
    embedding = session.run(None, {input_name: face})[0]

    # Normalizzazione L2
    embedding = embedding / np.linalg.norm(embedding)

    return embedding.flatten()

# ---- TEST ----
# Carica un'immagine reale
img = cv2.imread("volto.jpg")  # metti qui il tuo file

embedding = get_embedding(img)
print("Embedding generato!")
print(embedding)
print("Dimensione:", embedding.shape)

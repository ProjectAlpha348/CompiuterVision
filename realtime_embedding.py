import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

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
# 2) Mediapipe Face Detection
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
                    embedding = get_embedding(face_crop)
                    print("\nEmbedding (128 valori):")
                    print(embedding)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.imshow("Webcam + Embedding", frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()

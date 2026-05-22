import cv2
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
mp_face = mp.solutions.face_detection

# Apri webcam
cap = cv2.VideoCapture(0)

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Converti in RGB per Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if results.detections:
            for detection in results.detections:
                # Bounding box normalizzato
                box = detection.location_data.relative_bounding_box
                x1 = int(box.xmin * w)
                y1 = int(box.ymin * h)
                x2 = int((box.xmin + box.width) * w)
                y2 = int((box.ymin + box.height) * h)

                # Ritaglia il volto
                face_crop = frame[y1:y2, x1:x2]

                # Mostra il volto ritagliato
                if face_crop.size > 0:
                    cv2.imshow("Volto Ritagliato", face_crop)

                # Disegna il box sul frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

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
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):

                # Colori diversi per ogni volto
                color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)][face_id % 5]

                # Disegno contorni
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=color, thickness=1)
                )

        cv2.imshow("Face Mesh Avanzato", frame)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()

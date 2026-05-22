import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Landmark che useremo per la pose
LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye": 263,
    "right_eye": 33,
    "mouth_left": 61,
    "mouth_right": 291
}

cap = cv2.VideoCapture(1)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
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
            face_landmarks = results.multi_face_landmarks[0]

            # 2D points (pixel)
            image_points = []
            for name, idx in LANDMARK_IDS.items():
                lm = face_landmarks.landmark[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                image_points.append((x, y))
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            image_points = np.array(image_points, dtype="double")

            # 3D model points (in mm, modello semplificato)
            model_points = np.array([
                (0.0, 0.0, 0.0),        # nose tip
                (0.0, -90.0, -10.0),    # chin
                (-60.0, 40.0, -30.0),   # left eye
                (60.0, 40.0, -30.0),    # right eye
                (-40.0, -50.0, -30.0),  # mouth left
                (40.0, -50.0, -30.0)    # mouth right
            ])

            # Matrice intrinseca della camera (approssimata)
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            dist_coeffs = np.zeros((4, 1))  # niente distorsione

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # Disegniamo un asse che esce dal naso
                nose_end_3d = np.array([[0, 0, 200.0]])
                nose_end_2d, _ = cv2.projectPoints(
                    nose_end_3d,
                    rotation_vector,
                    translation_vector,
                    camera_matrix,
                    dist_coeffs
                )

                p1 = (int(image_points[0][0]), int(image_points[0][1]))  # nose tip
                p2 = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))
                cv2.line(frame, p1, p2, (0, 0, 255), 2)

                # Convertiamo in angoli (yaw, pitch, roll)
                rmat, _ = cv2.Rodrigues(rotation_vector)
                sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
                singular = sy < 1e-6

                if not singular:
                    pitch = np.arctan2(rmat[2,1], rmat[2,2])
                    yaw   = np.arctan2(-rmat[2,0], sy)
                    roll  = np.arctan2(rmat[1,0], rmat[0,0])
                else:
                    pitch = np.arctan2(-rmat[1,2], rmat[1,1])
                    yaw   = np.arctan2(-rmat[2,0], sy)
                    roll  = 0

                # Convertiamo in gradi
                pitch_deg = pitch * 180.0 / np.pi
                yaw_deg   = yaw   * 180.0 / np.pi
                roll_deg  = roll  * 180.0 / np.pi

                text = f"Yaw: {yaw_deg:5.1f}  Pitch: {pitch_deg:5.1f}  Roll: {roll_deg:5.1f}"
                cv2.putText(frame, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Head Pose Estimation", frame)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()

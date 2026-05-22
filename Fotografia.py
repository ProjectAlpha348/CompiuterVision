import cv2

# Apri la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Errore: impossibile aprire la webcam")
    exit()

print("Premi SPAZIO per scattare la foto, ESC per uscire")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Errore nel leggere il frame")
        break

    # Mostra l'immagine della webcam
    cv2.imshow("Webcam - premi SPAZIO per scattare", frame)

    key = cv2.waitKey(1)

    # Tasto SPAZIO → salva la foto
    if key == 32:  # codice ASCII dello spazio
        cv2.imwrite("volto.jpg", frame)
        print("Foto salvata come volto.jpg")
        break

    # Tasto ESC → esci
    if key == 27:
        print("Uscita senza salvare")
        break

cap.release()
cv2.destroyAllWindows()

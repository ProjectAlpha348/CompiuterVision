import cv2

def test_resolutions(cap, resolutions):
    results = []
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        results.append((w, h, real_w, real_h))
    return results

def test_parameters(cap, params):
    results = {}
    for name, prop in params.items():
        value = cap.get(prop)
        results[name] = value
    return results

def test_modifiable(cap, params):
    modifiable = {}
    for name, prop in params.items():
        original = cap.get(prop)
        if original == -1:
            modifiable[name] = False
            continue

        cap.set(prop, original + 1)
        new_value = cap.get(prop)

        modifiable[name] = (new_value != original)

        cap.set(prop, original)
    return modifiable

# --- MAIN ---

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Errore: impossibile aprire la webcam.")
    exit()

resolutions = [
    (1920,1080), (1600,900), (1280,720),
    (1024,576), (800,600), (640,480)
]

params = {
    "BRIGHTNESS": cv2.CAP_PROP_BRIGHTNESS,
    "CONTRAST": cv2.CAP_PROP_CONTRAST,
    "SATURATION": cv2.CAP_PROP_SATURATION,
    "HUE": cv2.CAP_PROP_HUE,
    "GAIN": cv2.CAP_PROP_GAIN,
    "EXPOSURE": cv2.CAP_PROP_EXPOSURE,
    "FOCUS": cv2.CAP_PROP_FOCUS,
    "ZOOM": cv2.CAP_PROP_ZOOM,
    "AUTOFOCUS": cv2.CAP_PROP_AUTOFOCUS,
}

print("\n--- TEST RISOLUZIONI ---")
res_results = test_resolutions(cap, resolutions)
for req_w, req_h, real_w, real_h in res_results:
    print(f"Richiesto {req_w}x{req_h} → Ottenuto {real_w}x{real_h}")

print("\n--- PARAMETRI DISPONIBILI ---")
param_results = test_parameters(cap, params)
for name, value in param_results.items():
    print(f"{name}: {value}")

print("\n--- PARAMETRI MODIFICABILI ---")
mod_results = test_modifiable(cap, params)
for name, can_modify in mod_results.items():
    print(f"{name}: {'Modificabile' if can_modify else 'Non modificabile'}")

cap.release()

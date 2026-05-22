import onnxruntime as ort

print("Caricamento modello...")

try:
    session = ort.InferenceSession("MobileFaceNet.onnx")
    print("Modello caricato correttamente!")

    # Mostra info sugli input
    print("\nInput del modello:")
    for inp in session.get_inputs():
        print(f" - {inp.name} : {inp.shape} ({inp.type})")

    # Mostra info sugli output
    print("\nOutput del modello:")
    for out in session.get_outputs():
        print(f" - {out.name} : {out.shape} ({out.type})")

except Exception as e:
    print("Errore durante il caricamento del modello:")
    print(e)

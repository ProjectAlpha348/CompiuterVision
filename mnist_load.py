import torch
from torchvision import datasets, transforms

# Trasformazione: converte l'immagine in tensore
transform = transforms.ToTensor()

# Carichiamo il dataset MNIST (train)
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

print("Numero di immagini nel dataset:", len(train_dataset))
print("Dimensione di una singola immagine:", train_dataset[0][0].shape)
print("Etichetta della prima immagine:", train_dataset[0][1])

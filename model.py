import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),            # 28x28 → 784
            nn.Linear(784, 128),     # primo livello
            nn.ReLU(),               # attivazione
            nn.Linear(128, 10)       # output: 10 classi
        )

    def forward(self, x):
        return self.model(x)

# Test rapido
net = SimpleNN()
x = torch.randn(1, 1, 28, 28)  # finta immagine
out = net(x)
print("Output shape:", out.shape)

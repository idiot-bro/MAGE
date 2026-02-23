import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)
class USAD(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder1 = Decoder(latent_dim, input_dim)
        self.decoder2 = Decoder(latent_dim, input_dim)

    def forward_ae1(self, x):
        z = self.encoder(x)
        return self.decoder1(z)

    def forward_ae2(self, x):
        z = self.encoder(x)
        return self.decoder2(z)

    def forward_ae2_on_ae1(self, x):
        z = self.encoder(x)
        x1 = self.decoder1(z)
        z1 = self.encoder(x1)
        return self.decoder2(z1)

def train_usad(model, dataloader, epochs=50, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x in dataloader:
            x = x.to(device)

            # AE1 & AE2 reconstruction
            x1 = model.forward_ae1(x)
            x2 = model.forward_ae2(x)

            # AE2(AE1(x))
            x12 = model.forward_ae2_on_ae1(x)


            loss_ae1 = (1 / epoch) * mse(x, x1) + (1 - 1 / epoch) * mse(x, x12)
            loss_ae2 = (1 / epoch) * mse(x, x2) - (1 - 1 / epoch) * mse(x, x12)

            loss = loss_ae1 + loss_ae2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch}/{epochs}], Loss: {total_loss:.4f}")

if __name__ == '__main__':
    pass



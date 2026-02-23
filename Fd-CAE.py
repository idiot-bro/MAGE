# Combining EEG Features and Convolutional Autoencoder for Neonatal Seizure Detection
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, input_len=198):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1  = nn.BatchNorm1d(16)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2  = nn.BatchNorm1d(32)
        self.enc3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3  = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2, 2)

        # Decoder
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.dec2 = nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1)
        self.dec1 = nn.ConvTranspose1d(16, 1,  kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, 1, L)
        e1 = F.relu(self.bn1(self.enc1(x))); p1 = self.pool(e1)
        e2 = F.relu(self.bn2(self.enc2(p1))); p2 = self.pool(e2)
        e3 = F.relu(self.bn3(self.enc3(p2))); p3 = self.pool(e3)
        # Decoder (mirror)
        d3 = self.up(p3); d3 = F.relu(self.dec3(d3))
        d2 = self.up(d3); d2 = F.relu(self.dec2(d2))
        d1 = self.up(d2); out = torch.sigmoid(self.dec1(d1))
        return out

    def encode(self, x):
        e1 = F.relu(self.bn1(self.enc1(x))); p1 = self.pool(e1)
        e2 = F.relu(self.bn2(self.enc2(p1))); p2 = self.pool(e2)
        e3 = F.relu(self.bn3(self.enc3(p2))); p3 = self.pool(e3)
        return p3  # latent feature map

class FCClassifier(nn.Module):
    def __init__(self, encoder, latent_dim, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(latent_dim, 64)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder.encode(x)
        z_flat = self.flatten(z)
        h = F.relu(self.fc1(z_flat))
        logits = self.fc2(h)
        return F.log_softmax(logits, dim=1)

if __name__ == '__main__':
    model = ConvAutoencoder(input_len=198)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, momentum=0.5)
    criterion = nn.MSELoss()
    for epoch in range(300):
        for x, _ in train_loader:
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()
    torch.save(model.state_dict(), 'cae.pth')

    cae = ConvAutoencoder(input_len=198)
    cae.load_state_dict(torch.load('cae.pth'))
    clf = FCClassifier(cae, latent_dim=64 * (198 // 8), num_classes=2)
    # 仅训练分类头
    optimizer = torch.optim.Adam(clf.fc1.parameters(), lr=1e-3)
    criterion = nn.NLLLoss()
    for epoch in range(100):
        for x, y in labeled_loader:
            logits = clf(x)
            loss = criterion(logits, y)
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()

# Unsupervised EEG-Based Seizure Anomaly Detection with Denoising Diffusion Probabilistic Models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes, code_dim):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1./num_codes, 1./num_codes)

    def forward(self, z_e):
        flat = z_e.permute(0,2,3,1).contiguous().view(-1, z_e.size(1))

        dists = (flat**2).sum(1, keepdim=True) - 2*flat @ self.codebook.weight.t() + \
                (self.codebook.weight**2).sum(1)
        idx = dists.argmin(1)
        z_q = self.codebook(idx).view(z_e.size(0), z_e.size(2), z_e.size(3), -1)
        z_q = z_q.permute(0,3,1,2)
        return z_q, idx

class VQVAE(nn.Module):
    def __init__(self, in_ch=1, hidden=256, num_codes=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 4, 2, 1), nn.ReLU()
        )
        self.quant = VectorQuantizer(num_codes, hidden)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(hidden, in_ch, 4, 2, 1)
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, idx = self.quant(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, z_q

class DDPM_UNet(nn.Module):
    def __init__(self, in_ch=256):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(in_ch, in_ch, 4, 2, 1), nn.ReLU()
        )
        self.up   = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(in_ch, in_ch, 3, padding=1)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, in_ch), nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, z_t, t):
        # z_t: [B, C, H, W], t: [B]
        h = self.down(z_t)
        emb = self.time_mlp(t.unsqueeze(-1).float()/1000.).view(-1, z_t.size(1), 1, 1)
        h = h + emb
        out = self.up(h)
        return out
class SAnoDDPM:
    def __init__(self, vqvae, ddpm, betas):
        self.vqvae = vqvae
        self.ddpm = ddpm
        self.register_betas(betas)

    def register_betas(self, betas):

        alphas = 1 - betas
        self.alpha_prod = torch.cumprod(torch.tensor(alphas), dim=0)

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_alpha = self.alpha_prod[t].sqrt().view(-1,1,1,1)
        sqrt_1ma   = (1-self.alpha_prod[t]).sqrt().view(-1,1,1,1)
        return sqrt_alpha*z0 + sqrt_1ma*noise

    def p_losses(self, z0, t):
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        eps_pred = self.ddpm(z_t, t)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def detect_and_recover(self, x, threshold, t_mask=500):
        _, z0 = self.vqvae(x)
        B = x.size(0)
        Ls = []
        for tau in range(400, 601):
            zt = self.q_sample(z0, torch.full((B,), tau, dtype=torch.long))
            eps_pred = self.ddpm(zt, torch.full((B,), tau, dtype=torch.long))
            Ls.append((zt - eps_pred).flatten(1).var(1))
        L_mean = torch.stack(Ls, dim=1).mean(1)
        mask = (L_mean > threshold).float().view(B,1,1,1)
        z_t = self.q_sample(z0, torch.full((B,), t_mask, dtype=torch.long))
        for step in reversed(range(t_mask)):
            eps_pred = self.ddpm(z_t, torch.full((B,), step, dtype=torch.long))
            z_t = (z_t - (1-self.alpha_prod[step]).sqrt()*eps_pred) / self.alpha_prod[step].sqrt()
            z_t = z_t*mask + z0*(1-mask)
        x_rec = self.vqvae.decoder(z_t)
        return mask, x_rec

    def train(self, loader, optimizer, epochs):
        self.vqvae.train(); self.ddpm.train()
        for ep in range(epochs):
            for x in loader:
                x = x.to(next(self.vqvae.parameters()).device)
                _, z0 = self.vqvae(x)
                t = torch.randint(0, len(self.alpha_prod), (x.size(0),), device=x.device)
                loss = self.p_losses(z0, t)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

if __name__ == '__main__':

    pass


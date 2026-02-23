import torch
import torch.nn as nn
import torch.nn.functional as F
import math

if __name__ == '__main__':
    class AnomalyAttention(nn.Module):
        """
        Anomaly-Attention:
        - Prior Association: Gaussian kernel (learnable sigma)
        - Series Association: Self-attention
        """

        def __init__(self, d_model, n_heads):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_head = d_model // n_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_sigma = nn.Linear(d_model, n_heads)

            self.out_proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            """
            x: [B, N, d_model]
            """
            B, N, _ = x.shape

            # Q K V
            Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
            K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
            V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

            # -------- Series Association --------
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
            series_attn = F.softmax(scores, dim=-1)  # [B, h, N, N]

            # -------- Prior Association --------
            sigma = self.W_sigma(x)  # [B, N, h]
            sigma = torch.exp(sigma).permute(0, 2, 1)  # [B, h, N]

            pos = torch.arange(N, device=x.device)
            dist = (pos[None, :] - pos[:, None]).abs().float()  # [N, N]

            prior_attn = []
            for i in range(self.n_heads):
                s = sigma[:, i, :].unsqueeze(-1)  # [B, N, 1]
                g = torch.exp(-dist ** 2 / (2 * s ** 2 + 1e-6))
                g = g / (g.sum(dim=-1, keepdim=True) + 1e-6)
                prior_attn.append(g)

            prior_attn = torch.stack(prior_attn, dim=1)  # [B, h, N, N]

            # -------- Value aggregation --------
            out = torch.matmul(series_attn, V)  # [B, h, N, d_head]
            out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
            out = self.out_proj(out)

            return out, prior_attn, series_attn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = AnomalyAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, prior, series = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, prior, series
class AnomalyTransformer(nn.Module):
    def __init__(self, input_dim, d_model=512, n_heads=8, n_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])

        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, x):
        """
        x: [B, N, input_dim]
        """
        x = self.embedding(x)

        priors, series = [], []
        for layer in self.layers:
            x, p, s = layer(x)
            priors.append(p)
            series.append(s)

        recon = self.decoder(x)
        return recon, priors, series

def association_discrepancy(priors, series):
    """
    Compute average symmetric KL divergence
    """
    loss = 0
    L = len(priors)

    for P, S in zip(priors, series):
        P = P + 1e-6
        S = S + 1e-6

        kl_ps = (P * (P.log() - S.log())).sum(dim=-1)
        kl_sp = (S * (S.log() - P.log())).sum(dim=-1)

        loss += (kl_ps + kl_sp).mean()

    return loss / L

def minimax_loss(x, recon, priors, series, lam=3.0, mode="max"):
    """
    mode = 'min' or 'max'
    """
    recon_loss = F.mse_loss(recon, x)

    if mode == "min":
        ass_dis = association_discrepancy(priors, [s.detach() for s in series])
        total_loss = recon_loss + lam * ass_dis

    else:  # maximize
        ass_dis = association_discrepancy(
            [p.detach() for p in priors], series
        )
        total_loss = recon_loss - lam * ass_dis

    return total_loss, recon_loss, ass_dis

def train_step(model, optimizer, x, lam=3.0):
    model.train()

    # -------- Minimize phase --------
    recon, priors, series = model(x)
    loss_min, _, _ = minimax_loss(
        x, recon, priors, series, lam, mode="min"
    )

    optimizer.zero_grad()
    loss_min.backward()
    optimizer.step()

    # -------- Maximize phase --------
    recon, priors, series = model(x)
    loss_max, _, _ = minimax_loss(
        x, recon, priors, series, lam, mode="max"
    )

    optimizer.zero_grad()
    loss_max.backward()
    optimizer.step()

    return loss_min.item(), loss_max.item()

def anomaly_score(x, recon, priors, series):
    rec_error = ((x - recon) ** 2).mean(dim=-1)
    ass_dis = association_discrepancy(priors, series)
    score = F.softmax(-ass_dis, dim=0) * rec_error
    return score
if __name__ == '__main__':
    B = 2  # batch size
    N = 100  # window length
    D = 3  # feature dimension

    data = torch.randn(B, N, D)
    recon, priors, series = model(data)



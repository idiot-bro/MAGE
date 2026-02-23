import torch
import torch.nn as nn
import torch.nn.functional as F


def mmd_loss(z, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute Maximum Mean Discrepancy (MMD) loss between encoded features z and N(0, I).
    """
    batch_size = z.size(0)
    # Compute pairwise distances
    z = z.view(batch_size, -1)
    XX = torch.matmul(z, z.t())  # [B, B]
    X2 = torch.sum(z * z, dim=1, keepdim=True)
    distances = X2 + X2.t() - 2 * XX

    # Kernel bandwidth
    if fix_sigma:
        sigma = fix_sigma
    else:
        sigma = torch.sum(distances.data) / (batch_size ** 2 - batch_size)
    bandwidth_list = [sigma * (kernel_mul ** i) for i in range(kernel_num)]
    kernels = [torch.exp(-distances / bw) for bw in bandwidth_list]
    kernel_sum = sum(kernels)

    # MMD loss
    XX = kernel_sum
    XY = kernel_sum
    YY = kernel_sum

    return torch.mean(XX + YY - 2 * XY)


class CAEM(nn.Module):
    def __init__(self, n_channels, time_steps, lambda_mmd=1e-4, lambda_pred=0.5):
        super(CAEM, self).__init__()
        self.lambda_mmd = lambda_mmd
        self.lambda_pred = lambda_pred

        # Characterization Network (CAE)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Memory Network
        self.bilstm = nn.LSTM(input_size=(n_channels + 1), hidden_size=512,
                              num_layers=1, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(512 * 2, 1)
        self.fc_pred = nn.Linear(512 * 2, n_channels + 1)
        self.ar_fc = nn.Linear(n_channels + 1, n_channels + 1)

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.size()
        x_in = x.unsqueeze(1)  # [B, 1, C, T]

        # Encode and reconstruct
        h = self.encoder(x_in)  # [B, 64, C/4, T/4]
        x_hat = self.decoder(h)  # [B, 1, C, T]

        # Flatten features
        z_f = torch.flatten(h, start_dim=1)  # [B, F]
        # Reconstruction error per timestep (mean over channels)
        recon_err = torch.mean((x_in - x_hat) ** 2, dim=[1, 2, 3], keepdim=True)
        # Build sequence tensor: assume sliding window of H steps already embedded
        # Here for simplicity, replicate same z_f over H steps
        H = 1  # adjust as needed
        z_r = recon_err.view(B, 1, 1)
        z_seq = torch.cat([z_f.view(B, 1, -1), z_r], dim=2)  # [B, H, F+1]

        # BiLSTM + attention
        y_seq, _ = self.bilstm(z_seq)  # [B, H, 1024]
        e = self.attn_fc(torch.tanh(y_seq))  # [B, H, 1]
        alpha = F.softmax(e, dim=1)
        context = torch.sum(alpha * y_seq, dim=1)  # [B, 1024]

        nonlin_pred = self.fc_pred(context)  # [B, F+1]
        lin_pred = self.ar_fc(z_seq[:, -1, :])  # [B, F+1]

        return x_hat, z_f, z_r, nonlin_pred, lin_pred, z_seq

    def compute_loss(self, x, outputs, z_true_seq=None):
        x_hat, z_f, z_r, nonlin_pred, lin_pred, z_seq = outputs
        # Reconstruction loss
        l_recon = F.mse_loss(x_hat, x.unsqueeze(1))
        # MMD loss
        l_mmd = mmd_loss(z_f)
        # Prediction loss
        if z_true_seq is None:
            z_true = z_seq[:, -1, :]
        else:
            z_true = z_true_seq[:, -1, :]
        l_pred = F.mse_loss(nonlin_pred, z_true) + F.mse_loss(lin_pred, z_true)

        return l_recon + self.lambda_mmd * l_mmd + self.lambda_pred * l_pred


if __name__ == '__main__':

    pass


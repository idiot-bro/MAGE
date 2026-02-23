# ChronoNet: A Deep Recurrent Neural Network for Abnormal EEG Identification
import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_scales=[2,4,8], stride=2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=stride, padding=k//2)
            for k in kernel_scales
        ])
    def forward(self, x):
        # x: (B, C, T)
        outs = [conv(x) for conv in self.convs]
        return torch.cat(outs, dim=1)  # (B, out_channels * len(kernel_scales), T')

class DenseGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
    def forward(self, x):
        out, _ = self.gru(x)
        return out

class ChronoNet(nn.Module):
    def __init__(self, in_channels=22, conv_channels=32, gru_hidden=32,
                 kernel_scales=[2,4,8], num_conv_layers=3, num_gru_layers=4, num_classes=2):
        super().__init__()
        convs = []
        ch = in_channels
        for _ in range(num_conv_layers):
            convs.append(InceptionConv1d(ch, conv_channels, kernel_scales))
            ch = conv_channels * len(kernel_scales)
        self.conv_seq = nn.Sequential(*convs)
        self.gru_layers = nn.ModuleList()
        self.gru_input_sizes = []
        for i in range(num_gru_layers):
            in_size = (i)*gru_hidden + ch if i>0 else ch
            self.gru_input_sizes.append(in_size)
            self.gru_layers.append(DenseGRULayer(in_size, gru_hidden))
        self.classifier = nn.Sequential(
            nn.Linear(num_gru_layers*gru_hidden, num_classes)
        )

    def forward(self, x):

        x = x.permute(0,2,1)
        x = self.conv_seq(x)
        x = x.permute(0,2,1)
        # Dense GRU
        feats = []
        for i, gru in enumerate(self.gru_layers):
            inp = torch.cat(feats + [x], dim=2) if i>0 else x
            out = gru(inp)  # (B, T', hidden)
            feats.append(out)
        last_steps = [f[:, -1, :] for f in feats]  # list of (B, hidden)
        h = torch.cat(last_steps, dim=1)           # (B, num_gru_layers*hidden)
        logits = self.classifier(h)
        return logits

if __name__ == "__main__":
    model = ChronoNet()
    dummy = torch.randn(8, 15000, 22)
    out = model(dummy)
    print("Output shape:", out.shape)




import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Using padding='same'.*")

class ImprovedLocalMemoryLayer(nn.Module):
    def __init__(self, mem_dim = 64, fea_dim=500, device='cpu', temperature=0.5):
        super(ImprovedLocalMemoryLayer, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.temperature = temperature
        self.std = 1.0 / math.sqrt(self.fea_dim)
    
        self.weight = nn.Parameter(
            torch.empty(self.fea_dim, self.mem_dim, device=device).uniform_(-self.std, self.std)
        ) # shape: (500, 64)
        self.gate_fc = nn.Sequential(
            nn.Linear(mem_dim, mem_dim // 3),
            nn.ReLU(),
            nn.Linear(mem_dim // 3, mem_dim)
        )
    def forward(self, inputs):
        batch_size, channel, x1_size, x2_size = inputs.shape
        #  [batch_size, x1_size, x2_size, channel]
        inputs_permuted = inputs.permute(0, 2, 3, 1)
        x = inputs_permuted.reshape(-1, channel)    # [batch_size * x1_size * x2_size, channel]
        distance = torch.matmul(x, self.weight.T)  # [batch_size * x1_size * x2_size = Tx, channel] * [channel, fea_dim]
        att_weight = F.softmax(distance / self.temperature, dim=1)  # [Tx, fea_dim]
        output = torch.matmul(att_weight, self.weight)  # [Tx, fea_dim] *[fea_dim, channel]

        output = output.view(batch_size, x1_size, x2_size, channel)
        output = output.permute(0, 3, 1, 2)

        pooled = F.adaptive_avg_pool2d(inputs, (1, 1)).view(batch_size, channel)
        gate = torch.sigmoid(self.gate_fc(pooled))  # [batch_size, channel],

        gate = gate.unsqueeze(-1).unsqueeze(-1)  # [batch_size, channel, 1, 1]

        output = gate * output + (1 - gate) * inputs

        att = att_weight.view(batch_size, x1_size * x2_size * self.fea_dim)
        entropy = -att * torch.log(att + 1e-12)
        att_entropy = entropy.mean(dim=-1, keepdim=True)  # [batch_size, 1]
        return output, att_entropy


class MultiHeadMemoryLayer(nn.Module):
    def __init__(self, mem_dim = 64, num_heads = 4, device='cpu', temperature=0.5):
        super(MultiHeadMemoryLayer, self).__init__()
        self.mem_dim = mem_dim
        self.num_heads = num_heads
        if mem_dim % self.num_heads != 0:
            raise ValueError(f"The input feature dimension must be divisible by num_heads: {num_heads}")
        self.head_dim = mem_dim // self.num_heads
        # self.temperature = temperature  #
        self.temperature = nn.Parameter(torch.tensor(temperature, device=device))
        self.std = 1.0 / math.sqrt(self.head_dim)
        #
        self.memory_keys = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim, self.mem_dim, device=device).uniform_(-self.std, self.std)
        ) # (num_heads, head_dim, mem_dim) (4, 16, 64)
        self.memory_values = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim, self.mem_dim, device=device).uniform_(-self.std, self.std)
        ) # (num_heads, head_dim, mem_dim) (4, 16, 64)

        self.gate_fc = nn.Sequential(
            nn.Linear(mem_dim, mem_dim // 3),
            nn.ReLU(),
            nn.Linear(mem_dim // 3, mem_dim),
            nn.BatchNorm1d(mem_dim)
        )
    def forward(self, inputs):
        batch, channel, x1_size, x2_size = inputs.shape
        x = inputs.permute(0, 2, 3, 1)  # [batch, x1_size, x2_size, channel] [B, H, W, C]
        #  [batch, x1_size, x2_size, num_heads, head_dim]
        x = x.view(batch, x1_size, x2_size, self.num_heads, self.head_dim)
        #  [B*H*W, num_heads, head_dim]
        x = x.reshape(-1, self.num_heads, self.head_dim)  #  [Tx, num_heads, head_dim]
        #
        distance = torch.einsum('bhd,hdm->bhm', x, self.memory_keys) # [Tx, num_heads, mem_dim]
        att_weight = F.softmax(distance / self.temperature, dim=-1)  # shape: [Tx, num_heads, mem_dim]
        # output shape: [Tx, num_heads, head_dim]
        output = torch.einsum('bhm,hdm->bhd', att_weight, self.memory_values)
        output = output.reshape(batch, x1_size, x2_size, self.num_heads * self.head_dim)
        output = output.permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(inputs, (1, 1)).view(batch, channel)
        gate = torch.sigmoid(self.gate_fc(pooled))  # [batch_size, channel], [0,1]

        gate = gate.unsqueeze(-1).unsqueeze(-1)  # [batch_size, channel, 1, 1]
        output = gate * output + (1 - gate) * inputs
        att_flat = att_weight.view(batch, x1_size * x2_size * self.num_heads, self.mem_dim)
        entropy = -att_flat * torch.log(att_flat + 1e-12)
        att_entropy = entropy.mean(dim=-1, keepdim=True).mean(dim=1)  # [batch, 1]
        return output, att_entropy

class MemoryLayer(nn.Module):
    def __init__(self, mem_dim, fea_dim = 500, device = 'cpu'):
        # mem_dim: size of the memory (M)
        # fea_dim: dimension of vector z (C)
        super(MemoryLayer, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.std = 1.0 / math.sqrt(self.fea_dim)
        # Memory weights (M x C)
        self.weight = nn.Parameter(torch.empty(self.fea_dim, self.mem_dim, device=device).uniform_(-self.std, self.std))

    def forward(self, inputs):
        # inputs shape: [batch_size, channel, x1_size, x2_size]
        inputs = inputs.permute(0, 2, 3, 1)
        # inputs shape: [batch_size, x1_size, x2_size, channel]
        batch_size, x1_size, x2_size, channel = inputs.shape # [10, 8, 128, 32]
        # Reshape inputs to [-1, channel]
        # inputs = inputs.view(-1, channel)
        inputs = inputs.reshape(-1, channel)
        # Compute distance (similarity) between inputs and memory
        # distance shape: [Tx, M] (Tx = batch_size * x1_size * x2_size)
        distance = torch.matmul(inputs, self.weight.T)  # (TxC) x (CxM) = TxM
        # Compute attention weights
        att_weight = F.softmax(distance, dim=1)  # Softmax along the memory dimension (M)
        # Compute output as weighted sum of memory
        # output shape: [Tx, C]
        output = torch.matmul(att_weight, self.weight)  # (TxM) x (MxC) = TxC
        # Reshape output back to [batch_size, x1_size, x2_size, channel]
        output = output.view(batch_size, x1_size, x2_size, channel)

        output = output.permute(0, 3, 1, 2)
        # Compute attention entropy
        # Flatten attention weights to [batch_size, x1_size * x2_size * fea_dim]
        att = att_weight.view(batch_size, x1_size * x2_size * self.fea_dim)
        att_weight = -att * torch.log(att + 1e-12)  # Avoid log(0) by adding a small constant
        att_weight = att_weight.mean(dim=-1, keepdim=True)  # Mean entropy per batch [batch, 1]

        return output, att_weight


class Encoder(nn.Module):
    def __init__(self, trans = 7, filter_size1 = 32, filter_size2 = 64):
        super(Encoder, self).__init__()
        self.trans = trans
        # Zero padding
        # self.zero_padding = nn.ZeroPad2d((3, 0, 3, 0))  # (left, right, top, bottom) (3, 3, 4, 3)
        # First Conv2D + MaxPool
        self.conv0 = nn.Conv2d(1, filter_size1, kernel_size=(4, 1))  # channels_last equivalent
        self.conv1 = nn.Sequential(
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        # Second Conv2D + MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter_size1, filter_size2, kernel_size=(4, 4), padding='same'),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
            nn.MaxPool2d(kernel_size=(2, 2))
        )
    def forward(self, inputs):
        ############################################################################################################################
        # Input shape: (batch_size, trans, 1, input_dim_x, input_dim_y)
        output = []
        for t in range(self.trans):
            x = self.conv0(inputs[:, t, :, :, :])
            # print(x.shape)
            x = self.conv1(x)
            x = self.conv2(x)
            output.append(x)
        output = torch.stack(output, dim=1)  # (batch_size, trans, filter_size2, height, width)
        return output


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        trans=7,
        in_channels=12,
        seq_len=257,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
        out_channels=64,
        out_h=63,
        out_w=3
    ):
        super(TransformerEncoder, self).__init__()

        self.trans = trans
        self.seq_len = seq_len
        self.out_w = out_w

        # ===============================
        # 1. Input projection
        # ===============================
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # ===============================
        # 2. Learnable positional embedding
        # ===============================
        self.pos_embed = nn.Parameter(
            torch.randn(1, seq_len, embed_dim)
        )

        # ===============================
        # 3. Transformer Encoder
        # ===============================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True   # Pre-LN
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ===============================
        # 4. Token Mixing (Depthwise Conv1D)
        # ===============================
        self.token_mixer = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim
        )

        # ===============================
        # 5. Channel Attention (SE-style)
        # ===============================
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(embed_dim, embed_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dim // 4, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # ===============================
        # 6. Output projection
        # ===============================
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, out_channels),
            nn.LayerNorm(out_channels)
        )

        # ===============================
        # 7. Height pooling (256 → 63)
        # ===============================
        self.pool_h = nn.AdaptiveAvgPool1d(out_h)

    def forward(self, inputs):
        """
        inputs: (B, trans=7, 1, 256, 19)
        return: (B, 7, 64, 63, 4)
        """
        B, T, _, H, C = inputs.shape
        outputs = []

        for t in range(T):
            # ===============================
            # (B, 256, 19)
            # ===============================
            x = inputs[:, t, 0]

            # ===============================
            # Input projection
            # (B, 256, embed_dim)
            # ===============================
            x = self.input_proj(x)

            # ===============================
            # Add positional embedding
            # ===============================
            x = x + self.pos_embed

            # ===============================
            # Transformer encoding
            # ===============================
            x = self.transformer(x)

            # ===============================
            # Token mixing
            # (B, embed_dim, 256)
            # ===============================
            x_mixed = self.token_mixer(x.permute(0, 2, 1))

            # ===============================
            # Channel attention
            # ===============================
            attn = self.channel_attn(x_mixed)
            x = x_mixed * attn + x_mixed   # residual

            # ===============================
            # Output projection
            # (B, 256, 64)
            # ===============================
            x = self.output_proj(x.permute(0, 2, 1))

            # ===============================
            # Pool to height = 63
            # (B, 64, 63)
            # ===============================
            x = self.pool_h(x.permute(0, 2, 1))

            # ===============================
            # Artificial width construction
            # (B, 64, 63, 4)
            # ===============================
            x = x.unsqueeze(-1).repeat(1, 1, 1, self.out_w)

            outputs.append(x)

        # ===============================
        # Stack over temporal dimension
        # (B, 7, 64, 63, 4)
        # ===============================
        return torch.stack(outputs, dim=1)
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        trans=7,
        in_channels=128,
        h_in=63,
        w_in=3,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
        out_h=257,
        out_w=12
    ):
        super(TransformerDecoder, self).__init__()

        self.trans = trans
        self.h_in = h_in
        self.w_in = w_in
        self.out_h = out_h
        self.out_w = out_w

        self.num_tokens = h_in * w_in

        # ===============================
        # 1. Memory token embedding
        # ===============================
        self.memory_proj = nn.Linear(in_channels, embed_dim)

        # ===============================
        # 2. Learnable decoder queries
        # ===============================
        self.query_embed = nn.Parameter(
            torch.randn(1, self.num_tokens, embed_dim)
        )

        # ===============================
        # 3. Positional embeddings
        # ===============================
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_tokens, embed_dim)
        )

        # ===============================
        # 4. Transformer Decoder
        # ===============================
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # ===============================
        # 5. Output projection
        # ===============================
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, in_channels),
            nn.LayerNorm(in_channels)
        )

        # ===============================
        # 6. Deconvolution blocks
        # ===============================
        self.deconvolution = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, 64,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=(1, 1)
                ),
                nn.BatchNorm2d(64),
                nn.GELU(),

                nn.ConvTranspose2d(
                    64, 32,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=(1, 1)
                ),
                nn.BatchNorm2d(32),
                nn.GELU(),

                nn.Conv2d(32, 1, kernel_size=3, padding=1)
            )
            for _ in range(trans)
        ])

    def forward(self, memory_output):
        """
        memory_output: (B, 7, 128, 63, 4)
        return:        (B, 7, 1, 257, 16)
        """
        B, T, C, H, W = memory_output.shape
        decoder_output = []

        for t in range(T):
            # ===============================
            # (B, 128, 63, 4) → tokens
            # ===============================
            x = memory_output[:, t]                       # (B, 128, 63, 4)
            x = x.flatten(2).permute(0, 2, 1)             # (B, 252, 128)

            # ===============================
            # Memory embedding
            # ===============================
            memory = self.memory_proj(x)                  # (B, 252, D)
            memory = memory + self.pos_embed

            # ===============================
            # Decoder queries
            # ===============================
            query = self.query_embed.expand(B, -1, -1)    # (B, 252, D)

            # ===============================
            # Transformer Decoder
            # ===============================
            decoded = self.transformer_decoder(
                tgt=query,
                memory=memory
            )                                              # (B, 252, D)

            # ===============================
            # Project back
            # ===============================
            decoded = self.output_proj(decoded)            # (B, 252, 128)

            # ===============================
            # Tokens → feature map
            # ===============================
            decoded = decoded.permute(0, 2, 1).reshape(
                B, C, H, W
            )                                              # (B, 128, 63, 4)

            # ===============================
            # Deconvolution
            # ===============================
            out = self.deconvolution[t](decoded)           # (~B, 1, 252, 16)

            # ===============================
            # Precise size alignment
            # ===============================
            out = F.interpolate(
                out,
                size=(self.out_h, self.out_w),
                mode="bilinear",
                align_corners=False
            )

            decoder_output.append(out)

        return torch.stack(decoder_output, dim=1)

    def mse_compute(self, encoder_input, decoder_output, dim = [1, 2, 3]):
        # TODO
        return torch.mean(torch.square(encoder_input - decoder_output))

    def mse_loss_compute(self, encoder_input, decoder_output):
        mse_loss_ = []
        for t in range(self.trans):
            mse_loss = self.mse_compute(encoder_input[:, t, :, :, :], decoder_output[:, t, :, :, :])
            mse_loss_.append(mse_loss)
        mse_loss = reduce(torch.add, mse_loss_)
        return mse_loss





class SelfSupervision(nn.Module):
    def __init__(self, filter_size2 = 64, trans = 7, dropout= 0.5, out_features = 128, fc_dim = 8 * 128, kernel_size = (4, 4)):
        super(SelfSupervision, self).__init__()
        self.trans = trans
        # Conv2D layer
        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels=filter_size2, out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=filter_size2, out_channels=1,  kernel_size=tuple(kernel_size)),
            nn.Sigmoid()
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, out_features),  #  is the flattened size after the convolution
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, trans)
        )

    def forward(self, inputs):
        # input shape: (batch, trans, filter_size2, 8, 128)
        output = []
        for t in range(self.trans):
            x = self.conv(inputs[:, t, :, :, :])
            x = torch.flatten(x, 1)
            x = self.fc(x)
            output.append(x)
        output = torch.stack(output, dim=1) # (batch, trans, trans)
        return output

class GlobalMemory(nn.Module):
    def __init__(self, trans, mem_dim, fea_dim, MultiHead = False, heads = 4, temperature = 0.5, device = 'cpu'):
        super(GlobalMemory, self).__init__()
        self.trans = trans
        if MultiHead:
            self.memory_global = MultiHeadMemoryLayer(mem_dim=mem_dim, num_heads=heads, temperature=temperature, device=device)
        else: self.memory_global = MemoryLayer(mem_dim = mem_dim, fea_dim = fea_dim, device=device)
    def forward(self, x):
        memory_output_g = []
        att_weight_g = []
        for t in range(self.trans):
            memory_output, att_weight = self.memory_global(x[:, t, :, :, :])
            memory_output_g.append(memory_output)
            att_weight_g.append(att_weight)
        memory_output_g = torch.stack(memory_output_g, dim=1)
        att_weight_g = torch.stack(att_weight_g, dim=1)
        return memory_output_g, att_weight_g
    def memory_global_sparse(self, att_weight_g):
        return att_weight_g.sum(dim=(1, 2))

class LocalMemory(nn.Module):
    def __init__(self, mem_dim, fea_dim, trans = 7, MultiHead = False, heads = 4, temperature = 0.5, device = 'cpu'):
        super(LocalMemory, self).__init__()
        self.trans = trans
        if MultiHead:
            self.memory_locals = nn.ModuleList([MultiHeadMemoryLayer(mem_dim = mem_dim, num_heads = heads, temperature=temperature, device=device) for _ in range(trans)])
        else: self.memory_locals = nn.ModuleList([MemoryLayer(mem_dim = mem_dim, fea_dim = fea_dim, device=device) for _ in range(trans)])
    def forward(self, x):
        memory_output_l = []
        att_weight_l = []
        for t in range(self.trans):
            memory_output, att_weight = self.memory_locals[t](x[:, t, :, :, :])
            memory_output_l.append(memory_output)
            att_weight_l.append(att_weight)
        memory_output_l = torch.stack(memory_output_l, dim=1)
        att_weight_l = torch.stack(att_weight_l, dim=1)
        return memory_output_l, att_weight_l
    def memory_local_sparse(self, att_weight_l):
        return att_weight_l.sum(dim=(1, 2))

class AdaptiveFusionLayer(nn.Module):
    def __init__(self, trans, momentum=0.93):
        super(AdaptiveFusionLayer, self).__init__()
        self.dense = nn.Linear(1, 2 * trans)
        self.batch_norm = nn.BatchNorm1d(2 * trans, momentum=momentum)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.dense(x)
        self.batch_norm.eval()
        x = self.batch_norm(x)
        x = self.sigmoid(x)
        return x

class AdaptiveFusion(nn.Module):
    def __init__(self, trans = 7, momentum=0.93):
        super(AdaptiveFusion, self).__init__()
        self.trans = trans
        self.adaptivefusion = AdaptiveFusionLayer(trans = trans, momentum=momentum)
    def forward(self, c, memory_output_l, memory_output_g):
        c = self.adaptivefusion(c)  # torch.Size([10, 14])
        m_out_1, m_out_2 = self.extract_m_out(c)
        final = self.calc_final(m_out_1, m_out_2, memory_output_l, memory_output_g)
        return final # list
    def extract_m_out(self, c):
        m_out_1, m_out_2 = [], []
        for i in range(0, 2 * self.trans, 2):
            m_out_1.append(c[:, i])
            m_out_2.append(c[:, i+1])
        return m_out_1, m_out_2
    def calc_final(self, m_out_1, m_out_2, memory_output_l, memory_output_g):
        final = []
        for i in range(self.trans):
            # Multiply
            local_weight = memory_output_l[:,i,:,:,:] * m_out_1[i].view(-1, 1, 1, 1)
            global_weight = memory_output_g[:,i,:,:,:] * m_out_2[i].view(-1, 1, 1, 1)
            # Add
            final.append(local_weight + global_weight)
        final = torch.stack(final, dim=1)
        return final

    def concatenate(self, x, final):
        memory_output = []
        for t in range(self.trans):
            memory_output.append(torch.cat((x[:, t, :, :, :], final[:, t, :, :, :]), dim=1))
        memory_output = torch.stack(memory_output, dim=1)
        return memory_output

class DeconvolutionLayer(nn.Module):
    def __init__(self,filter_size0, filter_size1, filter_size2, filter_size3, kernel_sizes, strides, paddings):
        super(DeconvolutionLayer, self).__init__()
        #(Conv2DTranspose)
        self.transpose_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filter_size3, out_channels=filter_size3, kernel_size=tuple(kernel_sizes[0]), stride=tuple(strides[0]), padding=paddings[0]),
            nn.GELU()
        )
        self.transpose_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filter_size3, out_channels=filter_size2, kernel_size=tuple(kernel_sizes[1]), stride=tuple(strides[1]), padding=paddings[1]),
            nn.GELU()
        )
        self.transpose_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filter_size2, out_channels=filter_size1, kernel_size=tuple(kernel_sizes[2]), stride=tuple(strides[2]), padding=paddings[2]),
            nn.GELU()
        )
        self.transpose_4 = nn.Sequential(
            # TODO DSADS kernel_size = (3, 3) TUSZ = (2, 5) SEED = (2, 4)
            nn.ConvTranspose2d(in_channels=filter_size1, out_channels=filter_size0, kernel_size=tuple(kernel_sizes[3]), stride=tuple(strides[3]), padding=paddings[3]),
            nn.Sigmoid()
        )
    def forward(self, memory_output_i):
        xx_ = self.transpose_1(memory_output_i) # [10, 128, 8, 128] -> [10, 128, 8, 128]            # [10, 128, 12, 32]
        xx_ = self.transpose_2(xx_)     # [10, 128, 8, 128] -> [10, 64, 16, 256]
        xx_ = self.transpose_3(xx_)     # [10, 64, 16, 256] -> [10, 32, 13, 253]
        xx_ = self.transpose_4(xx_)     # [10, 32, 13, 253] -> torch.Size([10, 1, 26, 506])
        output = xx_
        return output

class Decoder(nn.Module):
    def __init__(self, filter_size0, filter_size1, filter_size2, filter_size3, kernel_sizes, strides, paddings, trans=7):
        super(Decoder, self).__init__()
        self.deconvolution = nn.ModuleList([DeconvolutionLayer(filter_size0 = filter_size0, filter_size1=filter_size1, filter_size2=filter_size2, filter_size3=filter_size3,
                                                               kernel_sizes = kernel_sizes, strides = strides, paddings = paddings) for _ in range(trans)])
        self.trans = trans
    def forward(self, memory_output):
        decoder_output = []
        for t in range(self.trans):
            decoder_output.append(self.deconvolution[t](memory_output[:,t,:,:,:]))
        decoder_output = torch.stack(decoder_output, dim=1)
        return decoder_output
    def mse_compute(self, encoder_input, decoder_output, dim = [1, 2, 3]):
        return torch.mean(torch.square(encoder_input - decoder_output))  #

    def mse_loss_compute(self, encoder_input, decoder_output):
        mse_loss_ = []
        for t in range(self.trans):
            mse_loss = self.mse_compute(encoder_input[:, t, :, :, :], decoder_output[:, t, :, :, :])
            mse_loss_.append(mse_loss)
        mse_loss = reduce(torch.add, mse_loss_)
        return mse_loss



class MAGE(nn.Module):
    def __init__(self, filter_size0 = 1, filter_size1 = 32, filter_size2=64 , filter_size3 = 128, trans=7, dropout=0.5,
                 MultiHead = False, heads = 4, temperature = 0.5, ss_output_features = 128, ss_kernel_size = (4, 4),
                 decoder_kernel_sizes = ((4, 4), (3, 3), (3, 3), (2, 5)), strides = ((2, 2), (1, 1), (2, 2), (1, 1)), paddings = (1, 1, 1, 0),
                 global_mem_dim = 500, local_mem_dim = 500, fc_dim = 8*128, momentum = 0.93, lambda1 = 1.0, lambda2 = 0.0002, device='cpu'):
        super(MAGE, self).__init__()

        self.encoder = Encoder(trans, filter_size1, filter_size2)
        # self.encoder = TransformerEncoder()
        # self supervision
        self.selfsupervision = SelfSupervision(filter_size2 = filter_size2, trans = trans, dropout= dropout,
                                               out_features = ss_output_features, fc_dim = fc_dim, kernel_size=ss_kernel_size)
        self.globalmemory = GlobalMemory(mem_dim=filter_size2, fea_dim=global_mem_dim, MultiHead = MultiHead,
                                         heads=heads, temperature = temperature, trans = trans, device = device)
        self.localmemory = LocalMemory(mem_dim = filter_size2, fea_dim = local_mem_dim,  MultiHead = MultiHead,
                                       heads = heads, temperature = temperature, trans = trans, device = device)
        self.adaptivefusion = AdaptiveFusion(trans = trans,  momentum=momentum)
        self.decoder = Decoder(filter_size0 = filter_size0, filter_size1 = filter_size1, filter_size2 = filter_size2, filter_size3 = filter_size3,
                               kernel_sizes = decoder_kernel_sizes, strides = strides, paddings = paddings, trans=trans)
        # self.decoder = TransformerDecoder()
        self.trans = trans
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, inputs):
        # inputs shape: (batch, trans, 1, 19, 500)
        x = self.encoder(inputs)
        g = self.selfsupervision(x)
        memory_output_g, att_weight_g = self.globalmemory(x)
        memory_global_sparse = self.globalmemory.memory_global_sparse(att_weight_g)
        memory_output_l, att_weight_l = self.localmemory(x)
        memory_local_sparse = self.localmemory.memory_local_sparse(att_weight_l)

        c = torch.zeros(inputs.shape[0], 1, device=self.device)
        final = self.adaptivefusion(c, memory_output_l, memory_output_g)
        memory_output = self.adaptivefusion.concatenate(x, final)

        decoder_output = self.decoder(memory_output)
        mse_loss = self.decoder.mse_loss_compute(inputs, decoder_output)
        sparse_loss = (memory_global_sparse + memory_local_sparse).sum(dim=0)
        loss_g = self.loss_g_compute(g)
        total_loss = mse_loss.sum() + self.lambda2 * sparse_loss + self.lambda1 * loss_g
        return decoder_output, mse_loss, sparse_loss, loss_g, total_loss
    def loss_g_compute(self, g):
        n = g.shape[0]
        label = torch.tensor([i for i in range(self.trans) for _ in range(n)], device=self.device)
        # y_classes = F.one_hot(label, num_classes = self.trans).float()
        # y_i_ = [y_classes[n * i :n * (i + 1)] for i in range(self.trans)]   # (batch, 7)  [1. 0. 0. 0. 0. 0. 0.]
        y_i_ = label.reshape(self.trans, n) # (trans, batch)
        loss_g_ = []
        for i in range(self.trans):
            loss_g_.append(F.cross_entropy(g[:, i, :], y_i_[i]))
        loss_g = reduce(torch.add, loss_g_)
        return loss_g



if __name__ == '__main__':
    pass





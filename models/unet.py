import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ModeEmbedding(nn.Module):
    """
    Embeds normal mode eigenvectors and frequencies into query vectors for cross-attention
    """
    def __init__(self, max_atoms: int = 50, d_atom: int = 64, d_model: int = 256):
        super().__init__()
        self.max_atoms = max_atoms
        self.d_model = d_model

        # per atom MLP, operates on last dim (3,) and then broadcasts
        # over all others
        self.atom_mlp = nn.Sequential(
            nn.Linear(3, d_atom),
            nn.ReLU(inplace=True),
            nn.Linear(d_atom, d_model)
        )

        self.out_proj = nn.Linear(d_model + 1, d_model)

    def forward(self, V_vib, omegas, atom_mask):
        batch, _, max_modes = V_vib.shape

        x = V_vib.reshape(batch, self.max_atoms, 3, max_modes)
        x = x.permute(0, 3, 1, 2) # (batch, max_modes, max_atoms, 3)

        x = self.atom_mlp(x)

        mask = atom_mask.unsqueeze(1).unsqueeze(-1).float()
        x = (x * mask).sum(dim=2)
        n_atoms = mask.sum(dim=2)

        x  = x / n_atoms

        omegas_expanded = omegas.unsqueeze(-1)
        x = torch.cat([x, omegas_expanded], dim=-1)

        return self.out_proj(x)


class CrossAttentionBottleneck(nn.Module):
    """
    Cross attention between normal mode queries and bottlneck features. 
    Treated as a squence over the r-dimension.
    """
    def __init__(self, channels: int, d_model: int, n_R: int, n_heads: int = 8):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            kdim=d_model,
            vdim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(channels)

    def forward(self, bottleneck, mode_embeddings, mode_mask):
        kv = bottleneck.permute(0, 2, 1) # (batch, n_R, channels)

        attn_out, _ = self.attn(
            query=kv,
            key=mode_embeddings,
            value=mode_embeddings,
            key_padding_mask=~mode_mask
        )
        out = self.norm(kv + attn_out)
        return out.permute(0, 2, 1)


class PairDensityUNet(nn.Module):
    def __init__(self,
                 n_R: int = 1259,
                 features: list = [32, 64, 128, 256],
                 kernel_size: int = 3,
                 dropout: float = 0.0,
                 d_model: int = 256,
                 n_heads: int = 8,
                 max_atoms: int = 50,
                 max_modes: int = 144
    ):
        super().__init__()

        # encoder
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        ch = 1
        for f in features:
            self.encoder_blocks.append(ConvBlock(ch, f, kernel_size, dropout=dropout))
            ch = f
        
        # bottleneck
        bottleneck_channels = features[-1] * 2
        self.bottle_neck_conv = ConvBlock(features[-1], bottleneck_channels, kernel_size, dropout)

        # mode_embedding + cross-attention
        self.mode_embedding = ModeEmbedding(
            max_atoms=max_atoms,
            d_atom=64,
            d_model=d_model,
        )

        # n_R at the bottleneck is downsampled by 2^len(features)
        bottleneck_n_R = n_R // (2 ** len(features))
        self.cross_attention = CrossAttentionBottleneck(
            channels=bottleneck_channels,
            d_model=d_model,
            n_R=bottleneck_n_R,
            n_heads=n_heads,
        )

        # decoder
        self.decoder_up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_residual_projs = nn.ModuleList()

        ch = bottleneck_channels
        for f in reversed(features):
            self.decoder_up_convs.append(
                nn.ConvTranspose1d(ch, f, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(ConvBlock(f*2, f, kernel_size, dropout))
            self.decoder_residual_projs.append(nn.Conv1d(f*2, f, kernel_size=1))
            ch = f

        # final output
        self.final_conv = nn.Conv1d(features[0], 1, kernel_size=1)

    def forward(self, pd_limited, V_vib, omegas, atom_mask, mode_mask):
        x = pd_limited.unsqueeze(1)

        # encode
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # bottleneck
        x = self.bottle_neck_conv(x)

        # embed normal modes and apply cross attention
        mode_emb = self.mode_embedding(V_vib, omegas, atom_mask)
        x = self.cross_attention(x, mode_emb, mode_mask)

        # decode
        skip_connections = skip_connections[::-1]
        for upconv, block, res_proj, skip in zip(
            self.decoder_up_convs,
            self.decoder_blocks,
            self.decoder_residual_projs,
            skip_connections,
        ):
            x = upconv(x)
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x_cat = torch.cat([skip, x], dim=1)

            residual = res_proj(x_cat)
            x = block(x_cat) + residual

        return self.final_conv(x).squeeze(1)

if __name__ == "__main__":
    # sanity check
    batch = 4
    n_R = 1259
    max_atoms = 50
    max_modes = 144
    n_atoms = 20
    n_modes = 3 * n_atoms - 6

    model = PairDensityUNet(
        n_R=n_R,
        features=[32, 64, 128, 256],
        kernel_size=3,
        dropout=0.0,
        d_model=256,
        n_heads=8,
        max_atoms=max_atoms,
        max_modes=max_modes,
    )

    pd_limited = torch.randn(batch, n_R)
    V_vib = torch.zeros(batch, 3 * max_atoms, max_modes)
    V_vib[:, :3 * n_atoms, :n_modes] = torch.randn(batch, 3 * n_atoms, n_modes)

    omegas = torch.zeros(batch, max_modes)
    omegas[:, :n_modes] = torch.rand(batch, n_modes) * 0.1

    atom_mask = torch.zeros(batch, max_atoms, dtype=torch.bool)
    atom_mask[:, :n_atoms] = True

    mode_mask = torch.zeros(batch, max_modes, dtype=torch.bool)
    mode_mask[:, :n_modes] = True

    out = model(pd_limited, V_vib, omegas, atom_mask, mode_mask)
    print(f"Input: {pd_limited.shape}")
    print(f"Output: {out.shape}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")


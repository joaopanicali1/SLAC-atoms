import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierDeconvLayer(nn.Module):
    def __init__(self, q_max, n_r, dr):
        super().__init__()
        q = torch.fft.rfftfreq(n_r, d=dr) * 2 * torch.pi
        lorch = torch.sinc(q / q_max)
        self.register_buffer('window', lorch)
        self.log_lambda = nn.Parameter(torch.zeros(len(q)))
    
    def forward(self, x):
        # x has dim (batch, 1, n_r, n_t)
        X_q = torch.fft.rfft(x, dim=2) # FFT along r
        W = self.window.unsqueeze(-1)
        lam = torch.exp(self.log_lamda).unsqueeze(-1)
        X_clean = X_q * W_conj() / (W.abs()**2 + lam)
        return torch.fft_irfft(X_clean, n=x.shape[2], dim=2)
    
class QMaxEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, q_max):
        return self.mlp(q_max.unsqueeze(-1).float())
    
class ModeAttentionBottleneck(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadedAttention(d_model, n_heads, batch_first=True)
        self.freq_embed = nn.Linear(1, d_model)

    def forward(self, x_latent, omega_k):
        # x_latent (flattened bottleneck features) has shape (batch, seq_len, d_model) 
        # omega_k are known vibrational frequencies
        freq_tokens = self.freq_embed(omega_k.unsqueeze(-1))
        out, attn_weights = self.cross_attn(x_latent, freq_tokens, freq_tokens)
        return out, attn_weights
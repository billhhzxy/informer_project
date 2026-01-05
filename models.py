import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int,
        e_layers: int,
        dropout: float,
        pred_len: int,
    ):
        super().__init__()
        self.skip = nn.Linear(c_in, pred_len)
        self.lstm = nn.LSTM(
            input_size=c_in,
            hidden_size=d_model,
            num_layers=e_layers,
            dropout=dropout if e_layers > 1 else 0.0,
            batch_first=True,
        )
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x[:, -1, :])
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.projection(last) + skip


class ITransformerRegressor(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        pred_len: int,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by n_heads({n_heads})")
        self.skip = nn.Linear(c_in, pred_len)
        self.in_proj = nn.Linear(seq_len, d_model)
        self.token_emb = nn.Parameter(torch.zeros(1, c_in, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x[:, -1, :])
        tokens = x.transpose(1, 2)
        h = self.in_proj(tokens) + self.token_emb
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.projection(pooled) + skip


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 1")
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor):
        b, t, c = x.shape
        pad = (self.kernel_size - 1) // 2
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1)
        trend = trend.transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class FourierFilter(nn.Module):
    def __init__(self, top_k: int):
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = torch.fft.rfft(x, dim=1)
        mag = x_f.abs().mean(dim=(0, 2))
        k = min(self.top_k, int(mag.shape[0]))
        idx = torch.topk(mag, k=k, largest=True).indices
        mask = torch.zeros_like(mag, dtype=torch.bool)
        mask[idx] = True
        x_f = x_f * mask.view(1, -1, 1)
        return torch.fft.irfft(x_f, n=x.shape[1], dim=1)


class FEDformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, top_k: int, decomp_kernel: int):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size=decomp_kernel)
        self.filter = FourierFilter(top_k=top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomp(x)
        seasonal = self.filter(seasonal)
        x = self.norm1(x + self.dropout(seasonal + trend))
        y = self.ffn(x)
        return self.norm2(x + self.dropout(y))


class FEDformerRegressor(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        pred_len: int,
        top_k: int = 8,
        decomp_kernel: int = 3,
    ):
        super().__init__()
        self.skip = nn.Linear(c_in, pred_len)
        self.in_proj = nn.Linear(c_in, d_model)
        self.blocks = nn.ModuleList(
            [FEDformerBlock(d_model=d_model, d_ff=d_ff, dropout=dropout, top_k=top_k, decomp_kernel=decomp_kernel) for _ in range(e_layers)]
        )
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x[:, -1, :])
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        last = h[:, -1, :]
        return self.projection(last) + skip


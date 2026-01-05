import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.token_conv(x)
        return x.transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class FullAttention(nn.Module):
    def __init__(self, dropout: float = 0.0, scale: float | None = None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = scale

    def forward(self, q, k, v, attn_mask=None):
        b, h, l_q, d = q.shape
        scale = self.scale or (1.0 / math.sqrt(d))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn, v)
        return context, attn


class ProbAttention(nn.Module):
    def __init__(self, factor: int = 5, dropout: float = 0.0, scale: float | None = None):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)
        self.scale = scale

    def _prob_qk(self, q, k, sample_k: int, n_top: int):
        b, h, l_k, d = k.shape
        _, _, l_q, _ = q.shape

        index_sample = torch.randint(l_k, (l_q, sample_k), device=q.device)
        k_expand = k.unsqueeze(-3).expand(b, h, l_q, l_k, d)
        k_sample = k_expand[:, :, torch.arange(l_q, device=q.device).unsqueeze(1), index_sample, :]
        q_expand = q.unsqueeze(-2)
        qk_sample = torch.matmul(q_expand, k_sample.transpose(-2, -1)).squeeze(-2)

        m = qk_sample.max(dim=-1).values - qk_sample.mean(dim=-1)
        top = m.topk(n_top, sorted=False).indices
        return top

    def forward(self, q, k, v, attn_mask=None):
        b, h, l_q, d = q.shape
        _, _, l_k, _ = k.shape

        sample_k = min(l_k, int(self.factor * math.log(l_k + 1)))
        n_top = min(l_q, int(self.factor * math.log(l_q + 1)))
        top_q = self._prob_qk(q, k, sample_k=sample_k, n_top=n_top)

        q_reduce = q.gather(dim=2, index=top_q.unsqueeze(-1).expand(-1, -1, -1, d))
        scale = self.scale or (1.0 / math.sqrt(d))
        scores_top = torch.matmul(q_reduce, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            expanded_mask = attn_mask.gather(dim=2, index=top_q.unsqueeze(-1).expand(-1, -1, -1, attn_mask.size(-1)))
            scores_top = scores_top.masked_fill(expanded_mask, float("-inf"))

        attn = torch.softmax(scores_top, dim=-1)
        attn = self.dropout(attn)
        context_top = torch.matmul(attn, v)

        if attn_mask is None:
            context = v.mean(dim=2, keepdim=True).expand(b, h, l_q, v.size(-1)).clone()
        else:
            context = torch.zeros(b, h, l_q, v.size(-1), device=v.device, dtype=v.dtype)

        context.scatter_(
            dim=2,
            index=top_q.unsqueeze(-1).expand(-1, -1, -1, v.size(-1)),
            src=context_top,
        )
        return context, None


class AttentionLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by n_heads({n_heads})")
        self.attention = attention
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        b, l, _ = x.shape
        q = self.q_proj(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        context, attn = self.attention(q, k, v, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(b, l, self.n_heads * self.d_head)
        return self.out_proj(self.dropout(context)), attn


class ConvLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.down_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.down_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, attention: AttentionLayer, d_model: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor, attn_mask=None):
        new_x, attn = self.attention(x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        out = self.norm2(x + y)
        return out, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers: list[EncoderLayer], conv_layers: list[ConvLayer] | None, norm_layer: nn.Module | None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask=None):
        attns = []
        if self.conv_layers is None:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        else:
            for i, attn_layer in enumerate(self.attn_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
                if i < len(self.conv_layers):
                    x = self.conv_layers[i](x)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class InformerEncoderRegressor(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        factor: int,
        activation: str,
        pred_len: int,
    ):
        super().__init__()
        self.skip = nn.Linear(c_in, pred_len)
        self.enc_embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=dropout)

        attn_layers = []
        conv_layers = []
        for i in range(e_layers):
            attn = ProbAttention(factor=factor, dropout=dropout)
            attn_layer = AttentionLayer(attention=attn, d_model=d_model, n_heads=n_heads, dropout=dropout)
            attn_layers.append(
                EncoderLayer(attention=attn_layer, d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation)
            )
            if i != e_layers - 1:
                conv_layers.append(ConvLayer(d_model=d_model))

        self.encoder = Encoder(
            attn_layers=attn_layers,
            conv_layers=conv_layers if len(conv_layers) > 0 else None,
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x[:, -1, :])
        x = self.enc_embedding(x)
        enc_out, _ = self.encoder(x)
        last = enc_out[:, -1, :]
        out = self.projection(last) + skip
        return out

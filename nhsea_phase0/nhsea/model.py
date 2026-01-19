"""Minimal Transformer model with probe-layer logits and U-theta pathway."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    T: int = 64
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 4
    dropout: float = 0.0


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        logits_raw = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        logits = logits_raw
        if bias is not None:
            logits = logits + bias.unsqueeze(1)
        if attn_mask is not None:
            mask = attn_mask[:, None, None, :].to(dtype=logits.dtype)
            logits = logits.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out(out)
        return out, (logits_raw if return_logits else None)


class TransformerLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.ReLU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, logits = self.attn(x, attn_mask=attn_mask, bias=bias, return_logits=return_logits)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x, logits


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, probe_layer: int = 2) -> None:
        super().__init__()
        self.cfg = cfg
        self.probe_layer = probe_layer
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.T, cfg.d_model)
        self.layers = nn.ModuleList([TransformerLayer(cfg) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(cfg.d_model, 1)

        self.u_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.u_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        idx = torch.arange(cfg.T, dtype=torch.float32)
        D = (idx[None, :] - idx[:, None]) / float(cfg.T)
        self.register_buffer("drift_D", D, persistent=False)

    def set_num_classes(self, num_classes: int) -> None:
        self.head = nn.Linear(self.cfg.d_model, num_classes)

    def _u_pathway(self, H: torch.Tensor) -> torch.Tensor:
        q = self.u_q(H)
        k = self.u_k(H)
        return (q @ k.transpose(-2, -1)) / (self.cfg.d_model ** 0.5)

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        variant: str = "mechanism",
        alpha: float = 0.0,
        beta: float = 0.0,
        return_probe: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.embed(input_ids) + self.pos_embed(pos)[None, :, :]

        probe_logits = None
        probe_U = None

        for idx, layer in enumerate(self.layers, start=1):
            bias = None
            return_logits = False
            if idx == self.probe_layer:
                H = x
                U = self._u_pathway(H)
                D = self.drift_D[:T, :T].to(H).unsqueeze(0).expand(B, -1, -1)
                if variant == "mechanism":
                    S = U - U.transpose(-2, -1)
                    bias = alpha * D + beta * S
                elif variant == "symmetric_control":
                    S_dummy = U + U.transpose(-2, -1)
                    bias = alpha * D + beta * S_dummy
                elif variant == "no_injection":
                    bias = alpha * D
                elif variant == "no_drift":
                    S = U - U.transpose(-2, -1)
                    bias = beta * S
                else:
                    raise ValueError(f"Unknown variant: {variant}")
                return_logits = return_probe
                probe_U = U
            x, logits = layer(x, attn_mask=attn_mask, bias=bias, return_logits=return_logits)
            if idx == self.probe_layer and return_probe:
                probe_logits = logits

        pooled = x[:, -1, :]
        logits_cls = self.head(pooled)
        return logits_cls, probe_logits, probe_U

"""Minimal causal LM matching Sentinel paper shapes (SwiGLU FFN expand≈4, tied embed)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * rms


def _Norm(dim: int) -> nn.Module:
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(dim)
    return _RMSNorm(dim)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, expand: int = 4) -> None:
        super().__init__()
        hidden = (2 * dim * expand) // 3
        self.gate = nn.Linear(dim, hidden, bias=True)
        self.up = nn.Linear(dim, hidden, bias=True)
        self.down = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, use_checkpoint: bool) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.use_checkpoint = use_checkpoint
        self.norm1 = _Norm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm2 = _Norm(dim)
        self.ffn = SwiGLUFFN(dim)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return x + self.out(y)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm2(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(self._attn, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._ffn, x, use_reentrant=False)
            return x
        return self._ffn(self._attn(x))


class PaperLM(nn.Module):
    def __init__(
        self,
        vocab: int,
        dim: int,
        blocks: int,
        heads: int,
        max_pos: int,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_pos, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, heads, gradient_checkpointing) for _ in range(blocks)]
        )
        self.norm = _Norm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.tok.weight  # tie
        self.max_pos = max_pos

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.max_pos:
            raise ValueError("sequence longer than max_pos")
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.tok(input_ids) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        return F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_batch(
    vocab: int,
    batch: int,
    seq: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab, (batch, seq), device=device)
    y = torch.randint(0, vocab, (batch, seq), device=device)
    return x, y

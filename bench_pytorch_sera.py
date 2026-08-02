"""
PyTorch throughput bench matching Sentinel SERA demo settings 1:1.

Arch: vocab=1000, d=64, layers=2, heads=4, max_pos=256,
      SwiGLU hidden=(2*d*4)//3, RMSNorm, RoPE, full causal, LM head + bias
Train: batch=32, grad_accum=2, max_pack_cols=8192, length_bucket=32,
       flash SDPA, activation checkpointing, AMP fp16 without GradScaler
       (Sentinel: preferMixedPrecision on, lossScale off for d<256)
Data: 11993 synthetic examples (same train count as SERA materialize),
      lengths drawn to resemble short-doc LM; CE ignores pad.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# --- match main.cpp / CudaLanguageModel train ---
VOCAB = 1000
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
MAX_POS = 256
EXPAND_RATIO = 4
HIDDEN = (2 * D_MODEL * EXPAND_RATIO) // 3  # 170

BATCH_SIZE = 32
GRAD_ACCUM = 2
MAX_PACK_COLS = 8192
LENGTH_BUCKET_STEP = 32
EPOCHS = 2
TRAIN_EXAMPLES = 11993  # SERA materialize train count
LR = 1e-3
SEED = 42


def length_bucket(true_length: int, maximum_position_count: int = MAX_POS) -> int:
    if true_length <= 0:
        return min(LENGTH_BUCKET_STEP, maximum_position_count)
    true_length = min(true_length, maximum_position_count)
    bucket = ((true_length + LENGTH_BUCKET_STEP - 1) // LENGTH_BUCKET_STEP) * LENGTH_BUCKET_STEP
    return min(max(bucket, true_length), maximum_position_count)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q,k: [B, H, T, Dh] ; cos/sin: [T, Dh/2]
    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos.unsqueeze(0).unsqueeze(0) - x_odd * sin.unsqueeze(0).unsqueeze(0)
        out_odd = x_even * sin.unsqueeze(0).unsqueeze(0) + x_odd * cos.unsqueeze(0).unsqueeze(0)
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_pos: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        half = self.head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_table", freqs.cos(), persistent=False)
        self.register_buffer("sin_table", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        cos = self.cos_table[:t]
        sin = self.sin_table[:t]
        q, k = apply_rope(q, k, cos, sin)

        # Flash / mem-efficient SDPA when available
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(y)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=True)
        self.up = nn.Linear(d_model, hidden, bias=True)
        self.down = nn.Linear(hidden, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_pos: int, hidden: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_pos)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList(
            [Block(D_MODEL, N_HEADS, MAX_POS, HIDDEN) for _ in range(N_LAYERS)]
        )
        self.final_norm = RMSNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=True)
        self.use_checkpoint = True

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T]
        x = self.embed(token_ids)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


@dataclass
class Example:
    input_ids: list[int]
    target_ids: list[int]


def make_dataset(n: int, rng: torch.Generator) -> list[Example]:
    # Match SERA materialize: ~1.207M predictions / 11993 examples ≈ mean length ~101
    mean_len = 101.0
    lengths = torch.normal(mean=mean_len, std=55.0, size=(n,), generator=rng)
    lengths = lengths.round().clamp(2, MAX_POS).long()
    examples: list[Example] = []
    for length in lengths.tolist():
        tokens = torch.randint(0, VOCAB, (length + 1,), generator=rng).tolist()
        examples.append(Example(input_ids=tokens[:-1], target_ids=tokens[1:]))
    return examples


def pack_and_train(model: CausalLM, examples: list[Example], device: torch.device) -> tuple[float, dict]:
    order = sorted(range(len(examples)), key=lambda i: len(examples[i].input_ids))
    pack_window = max(BATCH_SIZE, MAX_PACK_COLS // LENGTH_BUCKET_STEP)
    examples_per_adam = BATCH_SIZE * GRAD_ACCUM

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    pack_count = 0
    single_packs = 0
    packed_ex_sum = 0
    packed_tok_sum = 0
    prediction_count = 0
    accum_examples = 0

    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    n = len(examples)
    for window_start in range(0, n, pack_window):
        window_end = min(window_start + pack_window, n)
        pack_start = window_start
        while pack_start < window_end:
            true_len = len(examples[order[pack_start]].input_ids)
            bucket = length_bucket(true_len)
            max_in_pack = max(1, MAX_PACK_COLS // bucket)

            pack_idx: list[int] = []
            pack_end = pack_start
            while pack_end < window_end and len(pack_idx) < max_in_pack:
                cand_len = len(examples[order[pack_end]].input_ids)
                if length_bucket(cand_len) != bucket:
                    break
                pack_idx.append(order[pack_end])
                pack_end += 1

            b = len(pack_idx)
            pack_count += 1
            if b <= 1:
                single_packs += 1
            packed_ex_sum += b
            packed_tok_sum += b * bucket

            inputs = torch.zeros(b, bucket, dtype=torch.long, device=device)
            targets = torch.full((b, bucket), -100, dtype=torch.long, device=device)
            for row, ei in enumerate(pack_idx):
                ex = examples[ei]
                L = len(ex.input_ids)
                inputs[row, :L] = torch.tensor(ex.input_ids, dtype=torch.long, device=device)
                targets[row, :L] = torch.tensor(ex.target_ids, dtype=torch.long, device=device)
                prediction_count += L

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(inputs)
                # per-example mean over true tokens, then mean over batch — close to Sentinel pack CE
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction="mean",
                )

            (loss / float(examples_per_adam) * float(b)).backward()
            # scale so that when accum hits examples_per_adam, effective mean is correct-ish
            accum_examples += b

            if accum_examples >= examples_per_adam or pack_end >= n:
                opt.step()
                opt.zero_grad(set_to_none=True)
                accum_examples = 0

            pack_start = pack_end

    if accum_examples > 0:
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    toks = prediction_count / sec if sec > 0 else 0.0
    stats = {
        "sec": sec,
        "predictions": prediction_count,
        "tok_s": toks,
        "packs": pack_count,
        "avgEx": packed_ex_sum / pack_count if pack_count else 0.0,
        "avgTok": packed_tok_sum / pack_count if pack_count else 0.0,
        "size1": 100.0 * single_packs / pack_count if pack_count else 0.0,
    }
    return toks, stats


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    rng = torch.Generator()
    rng.manual_seed(SEED)

    print("-- pytorch sera bench --")
    print(
        f"  config  vocab={VOCAB} d={D_MODEL} layers={N_LAYERS} heads={N_HEADS} "
        f"maxPos={MAX_POS} hidden={HIDDEN} maxPackCols={MAX_PACK_COLS} "
        f"batch={BATCH_SIZE} accum={GRAD_ACCUM} flash=sdpa checkpoint=on amp=fp16(no scaler)"
    )
    print(f"  torch   {torch.__version__}  cuda={torch.version.cuda}  device={torch.cuda.get_device_name(0)}")

    examples = make_dataset(TRAIN_EXAMPLES, rng)
    pred_total = sum(len(e.input_ids) for e in examples)
    print(f"  data    examples={len(examples)}  predictions/epoch={pred_total}")

    model = CausalLM().to(device)

    # warmup 1 small pack
    warm = torch.randint(0, VOCAB, (4, 64), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        model(warm).sum().backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    for epoch in range(EPOCHS):
        _, stats = pack_and_train(model, examples, device)
        print(
            f"  Epoch {epoch:<3d}  sec={stats['sec']:.2f}  tokens/s={stats['tok_s']:.0f}  backend=pytorch"
        )
        print(
            f"  pack  packs={stats['packs']}  avgEx={stats['avgEx']:.2f}  "
            f"avgTok={stats['avgTok']:.0f}  size1={stats['size1']:.1f}%"
        )


if __name__ == "__main__":
    main()

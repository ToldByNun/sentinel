"""Exercise mid-level layer / attention / FFN / SafeTensors bindings."""

from __future__ import annotations

import tempfile
from pathlib import Path

import sentinel as S


def main() -> None:
    embed_dim, heads, seq, pos = 32, 4, 8, 16

    # Standalone attention + SwiGLU
    attn = S.CausalSelfAttention.create(embed_dim, heads, pos, seed=1, kv_head_count=2)
    ffn = S.FeedForward.create_with_intermediate_size(embed_dim, 48, seed=2, use_bias=True)
    x = S.UniformInit.matrix(embed_dim, seq, 0.05, seed=3)

    attn_cache = S.CausalSelfAttentionCache()
    y = attn.forward(x, attn_cache)
    din, dq, dk, dv, do = attn.backward(y, attn_cache)
    print(f"attn out={y.shape} din={din.shape} dq={dq.shape}")

    ffn_cache = S.FeedForwardCache()
    h = ffn.forward(x, ffn_cache)
    grads = ffn.backward(h, ffn_cache)
    print(f"ffn out={h.shape} gate_w_grad={grads[1].shape}")

    # Host SPULSE on one matrix
    param = S.UniformInit.matrix(16, 16, 0.1, seed=4)
    grad = S.UniformInit.matrix(16, 16, 0.01, seed=5)
    state = S.SpulseState.zeros_like(param)
    opt = S.Spulse(learning_rate=1e-2)
    opt.step()
    opt.update(param, state, grad)
    print(f"spulse scale={state.scale:.4f} energy_fast={state.energy_fast:.6f}")

    # SafeTensors roundtrip
    st = S.SafeTensorsFile()
    st.metadata["format"] = "demo"
    st.put_matrix("w", param)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "w.safetensors")
        S.safetensors_save(path, st)
        loaded = S.safetensors_load(path)
        print(f"safetensors names={loaded.tensor_names} shape={loaded.get_tensor('w').shape}")

    # Block-level access on a tiny LM
    model = S.LanguageModel(64, embed_dim, pos, 1e-3, block_count=1, head_count=heads, kv_head_count=2)
    block = model.block(0)
    print(
        f"block attention heads={block.attention.head_count} "
        f"kv={block.attention.kv_head_count} "
        f"ffn_intermediate={block.feed_forward.intermediate_size}"
    )


if __name__ == "__main__":
    main()

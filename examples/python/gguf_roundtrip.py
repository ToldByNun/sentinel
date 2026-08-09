"""Export a tiny model to GGUF and reload it.

  python examples/python/gguf_roundtrip.py --out toy.gguf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sentinel as S


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("toy.gguf"))
    parser.add_argument("--architecture", default="llama", choices=("llama", "mistral", "qwen2"))
    args = parser.parse_args()

    model = S.LanguageModel(
        vocabulary_size=64,
        embedding_dim=32,
        maximum_position_count=64,
        learning_rate=3e-3,
        block_count=2,
        head_count=4,
        intermediate_size=64,
        rope_theta=10000.0,
        use_bias=False,
        kv_head_count=2,
    )
    model.set_tie_embedding(True)

    before = model.forward([1, 2, 3, 4])
    model.save_gguf(str(args.out), architecture=args.architecture)
    loaded = S.LanguageModel.load_gguf(str(args.out), learning_rate=3e-3)

    if loaded.kv_head_count != 2 or loaded.intermediate_size != 64 or loaded.use_bias:
        raise SystemExit("gguf_roundtrip: arch fields mismatch after reload")
    if not loaded.tie_embedding:
        raise SystemExit("gguf_roundtrip: expected tied embeddings")

    after = loaded.forward([1, 2, 3, 4])
    if after.rows != before.rows or after.cols != before.cols:
        raise SystemExit("gguf_roundtrip: logits shape mismatch")

    cfg = loaded.sentinel_config()
    print(
        f"gguf_roundtrip: wrote {args.out} arch={args.architecture} "
        f"vocab={cfg.vocab_size} embed={cfg.embedding_dim} "
        f"kv={loaded.kv_head_count} tie={loaded.tie_embedding}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

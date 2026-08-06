"""Train a small causal LM from a JSONL file and write weights.

JSONL lines may be:
  - {"text": "..."} / {"content": "..."} / {"problem_statement": "..."}
  - raw non-empty text (treated as the sample string)

Streaming ChunkSource (C++) is not wrapped in Python yet — this loads texts
into memory via LanguageModelDataset.build.

  python examples/python/train_jsonl.py examples/data/sample.jsonl --out run.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentinel as S


def load_texts(path: Path, max_rows: int) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            text = ""
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict):
                    for key in ("text", "content", "problem_statement"):
                        value = obj.get(key)
                        if isinstance(value, str) and value.strip():
                            text = value.strip()
                            break
            if not text:
                text = line
            texts.append(text)
            if max_rows > 0 and len(texts) >= max_rows:
                break
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Path to .jsonl corpus")
    parser.add_argument("--out", type=Path, default=Path("jsonl_demo.safetensors"))
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    if not args.jsonl.is_file():
        raise SystemExit(f"jsonl not found: {args.jsonl}")

    texts = load_texts(args.jsonl, args.max_rows)
    if len(texts) < 2:
        raise SystemExit("need at least 2 non-empty rows")

    tok = S.BPETokenizer()
    tok.train(texts, vocab_size=args.vocab_size)
    train = S.LanguageModelDataset.build(
        texts, tok, maximum_token_count=args.max_tokens, build_one_hot=False
    )
    if train.size == 0:
        raise SystemExit("no examples after tokenization (texts too short?)")

    max_pos = max(args.max_tokens, 32)
    model = S.LanguageModel(
        vocabulary_size=tok.vocab_size,
        embedding_dim=args.embed,
        maximum_position_count=max_pos,
        learning_rate=args.lr,
        block_count=args.blocks,
        head_count=args.heads,
    )

    if S.cuda_available():
        model.enable_cuda()
        model.set_prefer_flash_attention(True)
        model.enable_cuda_train()
        model.set_activation_checkpoint_mode(S.ActivationCheckpointMode.Off)
        print("train_jsonl: CUDA train enabled")
    else:
        print("train_jsonl: CUDA unavailable — host OpenMP train")

    model.train(train, epochs=args.epochs, batch_size=args.batch_size, log_every_epochs=max(1, args.epochs // 3))

    out = args.out
    if out.suffix.lower() == ".safetensors":
        model.save_safetensors(str(out))
    else:
        model.save_checkpoint(str(out), include_optimizer=False)

    prompt = texts[0][:40]
    ids = tok.encode(prompt)
    if ids:
        cont = model.generate(ids, new_token_count=24, temperature=0.9, top_k=20, seed=7)
        print(f"sample: {tok.decode(list(ids) + list(cont))}")

    print(
        f"train_jsonl: rows={len(texts)} examples={train.size} vocab={tok.vocab_size} "
        f"params~{model.parameter_count / 1e6:.3f}M avgLoss={model.average_loss(train):.4f} wrote {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

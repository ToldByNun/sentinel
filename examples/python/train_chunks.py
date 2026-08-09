"""Stream-train a small causal LM from JSONL / HF Arrow via LanguageModelChunkSource.

JSONL rows may use `problem_statement`, `text`, or `content` (same as JsonlLoader).

  python examples/python/train_chunks.py examples/data/sample.jsonl --out chunks_demo.snlm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sentinel as S


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Path to .jsonl or HF Arrow directory")
    parser.add_argument("--out", type=Path, default=Path("chunks_demo.snlm"))
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--embed", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-examples", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--test-cap", type=int, default=8)
    parser.add_argument("--sample-rows", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument(
        "--manual-chunks",
        action="store_true",
        help="Train via iter_train_chunks + train_step instead of train_chunks()",
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"corpus not found: {args.corpus}")

    source = S.LanguageModelChunkSource(
        str(args.corpus),
        maximum_text_characters=0,
        maximum_token_count=args.max_tokens,
        chunk_example_count=args.chunk_examples,
        train_ratio=args.train_ratio,
        seed=42,
        test_reservoir_cap=args.test_cap,
    )
    sample = source.prepare_tokenizer_sample(args.sample_rows)
    if len(sample) < 2:
        raise SystemExit("need at least 2 tokenizer sample rows (check JSONL text fields)")

    tok = S.BPETokenizer()
    tok.train(sample, vocab_size=args.vocab_size)
    source.set_tokenizer(tok)
    source.materialize()
    if source.train_example_count < 1:
        raise SystemExit("no train examples after materialize")

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
        print("train_chunks: CUDA train enabled")
    else:
        print("train_chunks: CUDA unavailable — host OpenMP train")

    if args.manual_chunks:
        for epoch in range(args.epochs):
            losses: list[float] = []
            for chunk in source.iter_train_chunks():
                losses.append(model.train_step(chunk.examples))
            mean = sum(losses) / max(1, len(losses))
            test_loss = (
                model.average_loss(source.test_dataset)
                if source.test_dataset.size > 0
                else float("nan")
            )
            print(
                f"  Epoch {epoch}  trainLoss={mean:.4f}  "
                f"chunks={len(losses)}  testLoss={test_loss:.4f}  backend=py-chunk-iter"
            )
    else:
        model.train_chunks(
            source,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            log_every_epochs=1,
        )

    out = args.out
    if out.suffix.lower() == ".safetensors":
        model.save_safetensors(str(out))
    else:
        model.save_checkpoint(str(out), include_optimizer=False)
    tok_path = out.with_suffix(".sbpe")
    tok.save(str(tok_path))

    prompt = sample[0][:40]
    ids = tok.encode(prompt)
    if ids:
        cont = model.generate(ids, new_token_count=16, temperature=0.9, top_k=20, seed=7)
        print(f"sample: {tok.decode(list(ids) + list(cont))}")

    chunk_count = sum(1 for _ in source.iter_train_chunks())
    print(
        f"train_chunks: train={source.train_example_count} test={source.test_dataset.size} "
        f"chunks/epoch={chunk_count} vocab={tok.vocab_size} "
        f"params~{model.parameter_count / 1e6:.3f}M wrote {out} + {tok_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

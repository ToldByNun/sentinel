from __future__ import annotations

import argparse
from pathlib import Path

import sentinel as S

CORPUS = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "a quick brown fox jumps",
    "sentinel trains causal language models",
    "cuda packs batches for throughput",
    "tiny models learn short patterns",
]


def sibling_sbpe(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".sbpe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="tiny_demo.snlm")
    parser.add_argument("prompt", nargs="?", default="the cat")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Path to .sbpe (default: sibling of checkpoint with .sbpe suffix)",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    tok_path = args.tokenizer if args.tokenizer is not None else sibling_sbpe(ckpt)

    if ckpt.exists() or Path(str(ckpt) + ".safetensors").exists():
        weight_path = str(ckpt) if ckpt.exists() else str(ckpt) + ".safetensors"
        if not tok_path.is_file():
            raise SystemExit(f"generate: missing tokenizer at {tok_path}")
        tok = S.BPETokenizer.load_from(str(tok_path))
        model = S.LanguageModel(tok.vocab_size, 64, 64, 3e-3, 2, 4)
        if S.cuda_available():
            model.enable_cuda()
        model.load_checkpoint(weight_path)
        print(f"generate: loaded {weight_path} + {tok_path}")
    else:
        print(f"generate: no checkpoint at {ckpt} — training toy model first")
        tok = S.BPETokenizer()
        tok.train(CORPUS, vocab_size=256)
        model = S.LanguageModel(tok.vocab_size, 64, 64, 3e-3, 2, 4)
        train = S.LanguageModelDataset.build(CORPUS, tok, maximum_token_count=48, build_one_hot=False)
        if S.cuda_available():
            model.enable_cuda()
            model.set_prefer_flash_attention(True)
            model.enable_cuda_train()
        model.train(train, epochs=8, batch_size=4, log_every_epochs=4)
        model.save_checkpoint(str(ckpt), include_optimizer=False)
        tok.save(str(tok_path))

    prompt_ids = tok.encode(args.prompt)
    if not prompt_ids:
        raise SystemExit("empty prompt encoding")

    continuation = model.generate(prompt_ids, new_token_count=32, temperature=0.9, top_k=20, seed=7)
    full = list(prompt_ids) + list(continuation)
    print(f"prompt: {args.prompt}")
    print(f"output: {tok.decode(full)}")


if __name__ == "__main__":
    main()

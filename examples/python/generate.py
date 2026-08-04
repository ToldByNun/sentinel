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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="tiny_demo.snlm")
    parser.add_argument("prompt", nargs="?", default="the cat")
    args = parser.parse_args()

    tok = S.BPETokenizer()
    tok.train(CORPUS, vocab_size=256)
    model = S.LanguageModel(tok.vocab_size, 64, 64, 3e-3, 2, 4)

    ckpt = Path(args.checkpoint)
    if ckpt.exists() or Path(str(ckpt) + ".safetensors").exists():
        path = str(ckpt) if ckpt.exists() else str(ckpt) + ".safetensors"
        if S.cuda_available():
            model.enable_cuda()
        model.load_checkpoint(path)
        print(f"generate: loaded {path}")
    else:
        print(f"generate: no checkpoint at {ckpt} — training toy model first")
        train = S.LanguageModelDataset.build(CORPUS, tok, maximum_token_count=48, build_one_hot=False)
        if S.cuda_available():
            model.enable_cuda()
            model.set_prefer_flash_attention(True)
            model.enable_cuda_train()
        model.train(train, epochs=8, batch_size=4, log_every_epochs=4)
        model.save_checkpoint(str(ckpt), include_optimizer=False)

    prompt_ids = tok.encode(args.prompt)
    if not prompt_ids:
        raise SystemExit("empty prompt encoding")

    continuation = model.generate(prompt_ids, new_token_count=32, temperature=0.9, top_k=20, seed=7)
    full = list(prompt_ids) + list(continuation)
    print(f"prompt: {args.prompt}")
    print(f"output: {tok.decode(full)}")


if __name__ == "__main__":
    main()

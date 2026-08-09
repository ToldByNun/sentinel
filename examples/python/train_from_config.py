"""Train a tiny model sized from a sentinel-model JSON/YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentinel as S


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(root / "configs" / "tiny.json"),
        help="sentinel-model JSON/YAML path (default: examples/configs/tiny.json)",
    )
    parser.add_argument("--out", default="tiny_from_config.snlm")
    args = parser.parse_args()

    corpus = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a quick brown fox jumps",
        "sentinel trains causal language models",
        "cuda packs batches for throughput",
        "tiny models learn short patterns",
    ]

    config = S.SentinelModelConfig.load(args.config)
    tok = S.BPETokenizer()
    tok.train(corpus, vocab_size=min(config.vocab_size, 256))
    # Rebuild shell with the trained vocab size; keep the rest of the config.
    config.vocab_size = tok.vocab_size
    config.weights = ""

    train = S.LanguageModelDataset.build(
        corpus, tok, maximum_token_count=config.max_position, build_one_hot=False
    )
    if train.size == 0:
        raise SystemExit("no examples after tokenization")

    model = S.LanguageModel.from_config(config, load_weights=False)

    if S.cuda_available():
        model.enable_cuda()
        model.set_prefer_flash_attention(True)
        model.enable_cuda_train()
        print("train_from_config: CUDA train enabled")
    else:
        print("train_from_config: CUDA unavailable — host OpenMP train")

    model.train(train, epochs=6, batch_size=4, log_every_epochs=2)
    model.save_checkpoint(args.out, include_optimizer=False)
    safe_path = args.out + ".safetensors"
    model.save_safetensors(safe_path)

    cfg_out = Path(args.out).with_suffix(".json")
    saved = model.sentinel_config()
    saved.weights = Path(safe_path).name
    saved.save(str(cfg_out))
    tok.save(str(Path(args.out).with_suffix(".sbpe")))

    print(
        f"train_from_config: examples={train.size} vocab={tok.vocab_size} "
        f"params~{model.parameter_count / 1e6:.4f}M "
        f"avgLoss={model.average_loss(train):.5f} "
        f"wrote {args.out} + {safe_path} + {cfg_out}"
    )


if __name__ == "__main__":
    main()

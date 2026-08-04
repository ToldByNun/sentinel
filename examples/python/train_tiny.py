from __future__ import annotations

import argparse

import sentinel as S


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="tiny_demo.snlm")
    args = parser.parse_args()

    corpus = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a quick brown fox jumps",
        "sentinel trains causal language models",
        "cuda packs batches for throughput",
        "tiny models learn short patterns",
    ]

    tok = S.BPETokenizer()
    tok.train(corpus, vocab_size=256)

    train = S.LanguageModelDataset.build(corpus, tok, maximum_token_count=48, build_one_hot=False)
    if train.size == 0:
        raise SystemExit("no examples after tokenization")

    model = S.LanguageModel(
        vocabulary_size=tok.vocab_size,
        embedding_dim=64,
        maximum_position_count=64,
        learning_rate=3e-3,
        block_count=2,
        head_count=4,
    )

    if S.cuda_available():
        model.enable_cuda()
        model.set_prefer_flash_attention(True)
        model.enable_cuda_train()
        model.set_activation_checkpoint_mode(S.ActivationCheckpointMode.Off)
        print("train_tiny: CUDA train enabled")
    else:
        print("train_tiny: CUDA unavailable — host OpenMP train")

    model.train(train, epochs=8, batch_size=4, log_every_epochs=2)
    model.save_checkpoint(args.checkpoint, include_optimizer=False)
    safe_path = args.checkpoint + ".safetensors"
    model.save_safetensors(safe_path)

    roundtrip = S.LanguageModel(tok.vocab_size, 64, 64, 3e-3, 2, 4)
    if S.cuda_available():
        roundtrip.enable_cuda()
    roundtrip.load_checkpoint(safe_path)

    print(
        f"train_tiny: examples={train.size} vocab={tok.vocab_size} "
        f"params~{model.parameter_count / 1e6:.4f}M "
        f"avgLoss={model.average_loss(train):.5f} "
        f"safeLoss={roundtrip.average_loss(train):.5f} "
        f"wrote {args.checkpoint} + {safe_path}"
    )


if __name__ == "__main__":
    main()

"""Custom host train loop via mid-level bindings (no LanguageModel.train)."""

from __future__ import annotations

import sentinel as S


def main() -> None:
    corpus = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "sentinel exposes matrix and adam ops",
        "custom loops call accumulate_example",
    ]
    tok = S.BPETokenizer()
    tok.train(corpus, vocab_size=128)
    data = S.LanguageModelDataset.build(
        corpus, tok, maximum_token_count=32, build_one_hot=False
    )

    model = S.LanguageModel(
        vocabulary_size=tok.vocab_size,
        embedding_dim=32,
        maximum_position_count=32,
        learning_rate=3e-3,
        block_count=2,
        head_count=4,
    )

    # Inspect / mutate tensors directly
    w = model.token_embedding.weight
    print(f"embed weight shape={w.shape} first={w.at(0, 0):.4f}")

    # Forward-only logits (optional CUDA mirror for inference)
    logits = model.forward(data[0].input_token_ids)
    print(f"logits shape={logits.shape}")

    # Manual Softmax + CE on one example
    probs = S.Softmax.apply(logits)
    target = S.LanguageModelDataset.make_one_hot_sequence(
        data[0].target_token_ids, tok.vocab_size
    )
    print(f"manual CE={S.CrossEntropy.loss(probs, target):.4f}")

    # Custom Adam microsteps
    for step in range(20):
        loss = model.train_step(data.examples)
        if step % 5 == 0 or step == 19:
            print(f"step={step:02d} loss={loss:.4f}")

    # Low-level equivalent of train_step for one example:
    grads = S.LanguageModelGradients.zeros_from(model)
    cache = S.LanguageModelCache()
    grads.zero_in_place()
    one = model.accumulate_example(data[0], grads, cache)
    grads.scale_in_place(1.0)
    model.apply_gradients(grads)
    print(f"single-example step loss={one:.4f} avg={model.average_loss(data):.4f}")


if __name__ == "__main__":
    main()

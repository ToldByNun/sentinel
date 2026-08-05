"""Fast repro: SERA-like cpuAdam + flash + test eval after epoch (the post-Epoch-0 crash path)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sentinel as S


def main() -> int:
    if not S.cuda_available():
        print("CUDA unavailable", flush=True)
        return 1

    # Match SERA shape (smaller data / fewer steps than full corpus).
    vocab_target = 512
    embed = 768
    blocks = 8
    heads = 12
    pos = 128
    seq = 64

    tok = S.BPETokenizer()
    corpus = [
        "hi",
        "ab",
        "the cat sat",
        "short",
        "a somewhat longer line of text for packing",
        "cuda stream train eval path repro",
    ] * 64
    tok.train(corpus, vocab_size=vocab_target)
    train = S.LanguageModelDataset.build(corpus, tok, maximum_token_count=seq)
    # Short + medium lengths like test reservoir noise
    test_corpus = ["hi", "x", "the cat", "a bit longer test sentence here"] * 32
    test = S.LanguageModelDataset.build(test_corpus, tok, maximum_token_count=seq)
    print(f"train={train.size} test={test.size} vocab~{tok.vocab_size}", flush=True)

    model = S.LanguageModel(tok.vocab_size, embed, pos, 3e-4, blocks, heads)
    model.enable_cuda()
    model.set_prefer_flash_attention(True)
    model.set_prefer_cpu_adam_offload(True)
    model.enable_cuda_train()
    model.set_prefer_train_graph(False)
    print(f"max_packed_columns={model.max_packed_columns}", flush=True)

    try:
        # This hits averageLoss(test) after epoch — same crash site as cuda-stream SERA.
        model.train(
            train,
            epochs=1,
            batch_size=32,
            gradient_accumulation_steps=2,
            log_every_epochs=1,
            test=test,
        )
        print("RESULT train+eval OK", flush=True)
        return 0
    except Exception as ex:
        print(f"RESULT FAILED: {ex}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Host-only CI smoke for the Python bindings (no GPU required).

Covers: import, BPE I/O, LanguageModel train/checkpoint/safetensors,
native sentinel-model config, mid-level Matrix ops, HF demo fine-tune export.

  python examples/python/ci_host_smoke.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import sentinel as S


def _expect(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"ci_host_smoke: {message}")


def test_import() -> None:
    _expect(isinstance(S.__version__, str) and len(S.__version__) > 0, "missing __version__")
    print(f"import ok version={S.__version__} cuda_available={S.cuda_available()}")


def test_matrix_ops() -> None:
    a = S.Matrix.from_list(2, 2, [1.0, 2.0, 3.0, 4.0])
    b = S.Matrix.scale(a, 0.5)
    _expect(abs(b.at(0, 0) - 0.5) < 1e-6, "Matrix.scale mismatch")
    probs = S.Softmax.apply(a)
    _expect(probs.rows == 2 and probs.cols == 2, "Softmax shape")
    print("matrix/softmax ok")


def test_bpe_and_tiny_train(tmp: Path) -> None:
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
    sbpe = tmp / "toy.sbpe"
    tok.save(str(sbpe))
    loaded = S.BPETokenizer.load_from(str(sbpe))
    _expect(loaded.encode("the cat") == tok.encode("the cat"), "BPE encode mismatch")
    _expect(loaded.is_trained, "BPE not trained after load")

    train = S.LanguageModelDataset.build(corpus, tok, maximum_token_count=48, build_one_hot=False)
    _expect(train.size > 0, "empty dataset")

    model = S.LanguageModel(
        vocabulary_size=tok.vocab_size,
        embedding_dim=64,
        maximum_position_count=64,
        learning_rate=3e-3,
        block_count=2,
        head_count=4,
    )
    # Host path only — CI runners have no GPU.
    model.train(train, epochs=2, batch_size=4, log_every_epochs=1)
    loss = model.average_loss(train)
    _expect(loss == loss and loss > 0.0, f"non-finite or non-positive loss: {loss}")

    snlm = tmp / "toy.snlm"
    safe = tmp / "toy.safetensors"
    model.save_checkpoint(str(snlm), include_optimizer=False)
    model.save_safetensors(str(safe))
    _expect(S.is_safetensors_file(str(safe)), "is_safetensors_file false")

    roundtrip = S.LanguageModel(tok.vocab_size, 64, 64, 3e-3, 2, 4)
    roundtrip.load_checkpoint(str(safe))
    safe_loss = roundtrip.average_loss(train)
    _expect(safe_loss == safe_loss, f"non-finite safetensors loss: {safe_loss}")

    ids = tok.encode("the")
    out = model.generate(ids, new_token_count=8, temperature=0.0, top_k=1, seed=1)
    _expect(len(out) == 8, f"generate length {len(out)}")
    print(f"tiny train ok loss={loss:.4f} safe_loss={safe_loss:.4f} params={model.parameter_count}")


def test_from_config(tmp: Path) -> None:
    cfg_path = Path("examples/configs/tiny.json")
    _expect(cfg_path.is_file(), f"missing {cfg_path}")
    model = S.LanguageModel.from_config(str(cfg_path), load_weights=False)
    out = tmp / "from_config.yaml"
    model.save_sentinel_config(str(out))
    _expect(out.is_file(), "save_sentinel_config missing file")
    snap = model.sentinel_config()
    _expect(snap.format == "sentinel-model", "bad format snapshot")
    _expect(snap.embedding_dim == 64, "unexpected embedding_dim from tiny.json")
    print("from_config ok")


def test_hf_demo_export(tmp: Path) -> None:
    # Lightweight stand-in for examples/python/finetune_hf.py --demo
    # Build a tiny allowlisted-shaped model, export HF dir, re-import.
    model = S.LanguageModel(
        vocabulary_size=128,
        embedding_dim=32,
        maximum_position_count=32,
        learning_rate=3e-3,
        block_count=1,
        head_count=4,
        intermediate_size=64,
        use_bias=False,
        kv_head_count=2,
    )
    export_dir = tmp / "hf_export"
    export_dir.mkdir()
    model.save_huggingface(str(export_dir), model_type="llama", weight_format="safetensors")
    _expect((export_dir / "config.json").is_file(), "HF export missing config.json")
    _expect((export_dir / "model.safetensors").is_file(), "HF export missing model.safetensors")

    loaded = S.LanguageModel.load_huggingface(str(export_dir), learning_rate=3e-3)
    _expect(loaded.kv_head_count == 2, "HF reimport kv_head_count")
    _expect(loaded.intermediate_size == 64, "HF reimport intermediate_size")
    _expect(not loaded.use_bias, "HF reimport use_bias")
    print("hf export/import ok")


def test_chunk_source_api() -> None:
    # Binding surface only — materialize needs problem_statement JSONL.
    _expect(hasattr(S, "LanguageModelChunkSource"), "LanguageModelChunkSource missing")
    _expect(hasattr(S.LanguageModel, "train_chunks"), "train_chunks missing")
    _expect(hasattr(S, "JsonlLoader"), "JsonlLoader missing")
    print("streaming bindings present")


def main() -> None:
    test_import()
    test_matrix_ops()
    test_chunk_source_api()
    with tempfile.TemporaryDirectory(prefix="sentinel_ci_") as tmp_name:
        tmp = Path(tmp_name)
        test_bpe_and_tiny_train(tmp)
        test_from_config(tmp)
        test_hf_demo_export(tmp)
    print("ci_host_smoke: all checks passed")


if __name__ == "__main__":
    main()

"""Fine-tune an allowlisted HuggingFace causal LM and export HF-compatible weights.

Supports a local HF model directory (config.json + safetensors / pytorch_model.bin).
If `huggingface_hub` is installed, `--model` may also be a Hub repo id or URL
(downloads into the standard HF cache — no Sentinel-owned cache).

  # Real checkpoint (local or Hub id)
  python examples/python/finetune_hf.py /path/to/hf_model examples/data/sample.jsonl \\
      --out hf_finetuned

  python examples/python/finetune_hf.py meta-llama/Llama-3.2-1B examples/data/sample.jsonl \\
      --out hf_finetuned --host-sgd   # needs HF_TOKEN for gated repos

  # Offline smoke: tiny stub model + tokenizer, no Hub download
  python examples/python/finetune_hf.py --demo --out hf_demo_out
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
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


def resolve_model_source(source: str) -> str:
    """Local HF dir, or Hub repo id / URL via huggingface_hub when installed."""
    path = Path(source)
    if path.is_dir() and (path / "config.json").is_file():
        return str(path.resolve())
    if path.is_file() and path.name == "config.json":
        return str(path.parent.resolve())

    resolve = getattr(S, "resolve_huggingface", None)
    if callable(resolve):
        return str(resolve(source))

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            f"model source is not a local HF directory with config.json: {source!r}\n"
            "Pass a local path, or install huggingface_hub "
            "(pip install 'sentinel-lm[hf]' / huggingface_hub) for Hub repo ids / URLs."
        ) from exc

    # Strip common HF URL prefixes so snapshot_download gets org/name.
    repo = source.strip()
    for prefix in (
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://hf.co/",
        "http://hf.co/",
    ):
        if repo.lower().startswith(prefix):
            repo = repo[len(prefix) :]
            break
    repo = repo.split("?")[0].split("#")[0].rstrip("/")
    for marker in ("/tree/", "/resolve/", "/blob/"):
        if marker in repo:
            repo = repo.split(marker, 1)[0]
            break
    if repo.startswith("models/"):
        repo = repo[len("models/") :]

    print(f"finetune_hf: downloading Hub snapshot for {repo!r} (standard HF cache)")
    return str(snapshot_download(repo_id=repo))


def write_demo_stub(directory: Path) -> list[str]:
    """Tiny Llama-shaped HF dir (weights + ByteLevel tokenizer) for offline demos."""
    directory.mkdir(parents=True, exist_ok=True)

    vocab_size = 32
    embed = 16
    model = S.LanguageModel(
        vocabulary_size=vocab_size,
        embedding_dim=embed,
        maximum_position_count=64,
        learning_rate=3e-4,
        block_count=2,
        head_count=4,
        intermediate_size=32,
        rope_theta=10000.0,
        use_bias=False,
        kv_head_count=2,
    )

    model.save_huggingface(
        str(directory),
        model_type="llama",
        tokenizer_source_directory="",
        weight_format="safetensors",
    )

    # Minimal ByteLevel BPE (same shape as the HF roundtrip smoke tokenizer).
    tokenizer_json = {
        "version": "1.0",
        "added_tokens": [
            {"id": 30, "content": "<bos>", "special": True},
            {"id": 31, "content": "<eos>", "special": True},
        ],
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "unk_token": None,
            "ignore_merges": False,
            "vocab": {
                "h": 0,
                "i": 1,
                "a": 2,
                "b": 3,
                "hi": 4,
                "ab": 5,
                "hiab": 6,
                "<bos>": 30,
                "<eos>": 31,
            },
            "merges": ["h i", "a b", "hi ab"],
        },
    }
    (directory / "tokenizer.json").write_text(
        json.dumps(tokenizer_json, indent=2), encoding="utf-8"
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}, indent=2),
        encoding="utf-8",
    )

    # Short strings the stub tokenizer can encode (≥2 tokens with BOS).
    return ["hi", "ab", "hiab", "hi", "ab", "hiab"]


def configure_device(model: S.LanguageModel, *, host_sgd: bool) -> None:
    if not S.cuda_available():
        print("finetune_hf: CUDA unavailable — host OpenMP train")
        return
    model.enable_cuda()
    model.set_prefer_flash_attention(True)
    if host_sgd:
        model.set_prefer_host_sgd(True)
        model.set_activation_checkpoint_mode(S.ActivationCheckpointMode.Full)
        print("finetune_hf: CUDA train + HostSGD + activation ckpt Full")
    else:
        model.set_activation_checkpoint_mode(S.ActivationCheckpointMode.Off)
        print("finetune_hf: CUDA train enabled")
    model.enable_cuda_train()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="",
        help="Local HF model directory, Hub repo id, or huggingface.co URL",
    )
    parser.add_argument(
        "jsonl",
        nargs="?",
        type=Path,
        default=None,
        help="JSONL corpus (default: examples/data/sample.jsonl; ignored with --demo)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Offline stub model+tokenizer (no Hub); trains a tiny built-in corpus",
    )
    parser.add_argument("--out", type=Path, default=Path("hf_finetuned"))
    parser.add_argument("--model-type", default="llama", choices=("llama", "mistral", "qwen2"))
    parser.add_argument(
        "--weight-format",
        default="safetensors",
        choices=("safetensors", "bin", "both"),
        help="Export weight file(s) for save_huggingface",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    parser.add_argument(
        "--host-sgd",
        action="store_true",
        help="Prefer HostSGD + full activation checkpointing (larger models / 16GB)",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional generate() prompt after train (empty → first corpus line)",
    )
    parser.add_argument("--new-tokens", type=int, default=24)
    args = parser.parse_args()

    demo_tmpdir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.demo:
            demo_tmpdir = tempfile.TemporaryDirectory(prefix="sentinel_hf_demo_")
            model_dir = Path(demo_tmpdir.name) / "stub"
            texts = write_demo_stub(model_dir)
            model_source = str(model_dir)
            print(f"finetune_hf: wrote offline demo stub under {model_dir}")
        else:
            if not args.model:
                raise SystemExit("pass a model path/repo id, or use --demo")
            model_source = resolve_model_source(args.model)
            jsonl = args.jsonl
            if jsonl is None:
                jsonl = Path(__file__).resolve().parents[1] / "data" / "sample.jsonl"
            if not jsonl.is_file():
                raise SystemExit(f"jsonl not found: {jsonl}")
            texts = load_texts(jsonl, args.max_rows)

        if len(texts) < 2:
            raise SystemExit("need at least 2 non-empty training strings")

        print(f"finetune_hf: loading model from {model_source}")
        model = S.LanguageModel.load_huggingface(model_source, learning_rate=args.lr)
        tok = S.HfTokenizer.load(model_source)

        train = S.LanguageModelDataset.build(
            texts,
            tok,
            maximum_token_count=args.max_tokens,
            build_one_hot=False,
        )
        if train.size == 0:
            raise SystemExit(
                "no examples after tokenization "
                "(texts too short for this tokenizer, or empty encodings?)"
            )

        configure_device(model, host_sgd=args.host_sgd)
        model.train(
            train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            log_every_epochs=max(1, args.epochs // 2),
        )

        out = args.out
        if out.exists():
            shutil.rmtree(out) if out.is_dir() else out.unlink()
        out.mkdir(parents=True, exist_ok=True)

        model.save_huggingface(
            str(out),
            model_type=args.model_type,
            tokenizer_source_directory=model_source,
            weight_format=args.weight_format,
        )

        prompt = args.prompt.strip() or texts[0]
        ids = tok.encode(prompt, add_special_tokens=True)
        if ids:
            cont = model.generate(
                ids,
                new_token_count=args.new_tokens,
                temperature=0.9,
                top_k=20,
                seed=7,
            )
            print(f"sample: {tok.decode(list(ids) + list(cont), skip_special_tokens=True)}")

        print(
            f"finetune_hf: texts={len(texts)} examples={train.size} "
            f"params~{model.parameter_count / 1e6:.3f}M "
            f"avgLoss={model.average_loss(train):.4f} "
            f"exported {out} (weight_format={args.weight_format})"
        )
        return 0
    finally:
        if demo_tmpdir is not None:
            demo_tmpdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

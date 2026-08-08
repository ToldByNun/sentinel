# Sentinel docs

Library reference for **v0.1**. High-level install and quickstarts live in the root [README](../README.md); this folder is the API contract.

| Page | Audience |
| ---- | -------- |
| [Python API](python.md) | `pip install sentinel-lm` → `import sentinel` |
| [C++ API](cpp.md) | CMake `find_package(Sentinel)` → `Sentinel::sentinel` |
| [HuggingFace import / export](huggingface.md) | HF causal-LM load/save + tokenizer + fine-tune notes |

**Not API:** `sentinel/main.cpp` smoke/scale harness flags, ad-hoc probe scripts unless linked from these pages.

```text
docs/
  README.md        ← you are here
  python.md        Python bindings
  cpp.md           C++ headers & patterns
  huggingface.md   HF checkpoint import
```

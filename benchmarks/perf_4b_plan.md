# Performance plan — 4B training toward ~1,300 tok/s

Goal: raise 4B packed-train throughput from the recorded **~950–1000 tok/s** to
**~1,300 tok/s** (≈ +30–37 %, i.e. cut step wall-time to ~0.73–0.77×) on the
`benchmarks/_lib/probe_4b_safe.py` setup.

This is a measurement-driven plan. Per [`CONTRIBUTING.md`](../CONTRIBUTING.md),
every train-path change needs a stated hypothesis and a multi-run mean tok/s
keep/revert gate — **stack only measured wins**.

## Baseline being optimized

`probe_4b_safe.py`: `vocab 32000 · embed 3072 · 34 blocks · 48 heads · pos 2048 ·
seq 512`, pack ≈ 8 → **4096 tokens/step**, SBAO `HostFusedHalfSgd`, **Full**
activation checkpointing, flash attention on, CUDA graphs off.
Reference: RTX 5070 Ti 16 GB, WDDM (`benchmarks/README.md`, root `README.md`
"Performance"). VRAM used ≈ 12.4 GiB (≈ 3.6 GiB headroom on 16 GB).

## Step 0 — Measure before touching code (mandatory)

1. Attribute the step budget with the built-in phase trace:
   `SENTINEL_PHASE_TRACE=1 python benchmarks/_lib/probe_4b_safe.py`
   → `phase accumulate: fwd=.. headCe=.. bwdGpu=.. d2h=.. sgd=.. h2d=..`
   (emitted at `sentinel/NeuralNet/Cuda/CudaLanguageModel.cu:620`).
2. Kernel-level: `benchmarks/_lib/profile_4b_nsys.py` (Nsight) for GEMM vs
   elementwise vs recompute occupancy, and to see exposed (non-overlapped) copies.
3. Quote tok/s only with `SENTINEL_PHASE_TRACE` **unset** (trace adds syncs).
   Use warm multi-run means; lock GPU clocks (or run on Linux/TCC, not WDDM) so
   the signal isn't swamped by desktop clock/pack variance.

The phase split decides how much of the budget below is real. The prioritization
assumes the dominant costs are (a) Full-checkpoint recompute and (b) the LM
head/CE, with (c) host-SGD copies partially exposed — **confirm with Step 0.**

## Implemented in this PR (opt-in; default = current behavior)

All three are gated so a plain run is byte-for-byte the shipped path — flip one
env var at a time, run the probe twice (with/without), and keep/revert per the
≥3 % mean gate.

| Lever | Env flag (default) | What it does | Where |
| --- | --- | --- | --- |
| **L4/L5 recompute** | `SENTINEL_CKPT_KEEP_FREE_MIB` (2048) | Lower the reserved-VRAM floor so `enablePartialSelectiveLayers` converts **more** trailing Full-ckpt layers to the cheap selective backward → fewer full-block recomputes. This is the biggest lever. | `CudaLanguageModel.cu` `enablePartialSelectiveLayers` |
| **L3 overlap depth** | `SENTINEL_MAX_ASYNC_HOST_UPDATE` (4) | Tune how many async host-SGD applies trail the GPU backward (deeper = more D2H/SGD/H2D hidden, if freelist slots allow). | `CudaLanguageModel.cu` backward loop |
| **L1 bias-GEMM algo cache (TF32)** | `SENTINEL_BIAS_GEMM_CACHE` (off) | Cache descriptor+algo for TF32 bias-epilogue GEMMs instead of re-running the heuristic per call. | `CudaMatmul.cu` `launchCublasLtMatmul` |

Also added: `benchmarks/_lib/autotune_pack_4b.py` — L2 pack-budget sweep that
reports tok/s per `max_packed_columns` for the 4B host-SGD (feat) path.

### Suggested A/B order on 4B_PoC feat (baseline 1.15k)
1. `SENTINEL_CKPT_KEEP_FREE_MIB=1024` (then 768, 512) — watch for OOM/WDDM thrash; expect the largest gain.
2. `python benchmarks/_lib/autotune_pack_4b.py --seq 256` — pick the best `max_packed_columns`.
3. `SENTINEL_MAX_ASYNC_HOST_UPDATE=6` (then 8) — only helps if Step 0's `d2h`/`sgd`/`h2d` are exposed.
4. `SENTINEL_BIAS_GEMM_CACHE=1` — **note:** the FP16 path (`CudaAmp.cu`) already caches the bias algo, so on the feat/AMP hot path this is expected to be ~neutral; it mainly helps non-AMP/TF32 GEMMs. Kept for completeness.

Combine the winners (they stack multiplicatively). Correctness note: every added
cuBLASLt cache path returns `false` on any failure, so the caller falls back to
its existing correct GEMM — a bad cache degrades speed, not numerics.

### Honest findings from the code (recalibration)
- **L1 was already done where it matters.** `CudaAmp.cu` (FP16 path used by 4B
  feat) already caches desc+algo keyed by shape+bias. The TF32 gap I closed is a
  minor/fallback win. Don't expect L1 to move 4B feat.
- **The recompute lever already exists** as `enablePartialSelectiveLayers`
  (VRAM-budgeted). It was hard-capped at a 2 GiB free floor; the new env lets you
  push it. This — plus pack budget — is where the +30 % realistically comes from.
- **Not shipped blind:** LM head/CE fusion (L6), a bf16 path (L7), and elementwise
  kernel fusion (L8). These are large and numerics-sensitive; writing them without
  a GPU to run the parity smokes would risk silent training corruption, which the
  repo's evidence-gate methodology (and basic sanity) forbids. They stay staged
  for a GPU-in-the-loop pass — see below.

## Prioritized levers (design detail / staged)

Each lever: hypothesis → where → expected delta → risk. Deltas are estimates to
be validated with the probe; keep only if the gate is met, else revert.

### Phase 1 — low risk, little/no extra VRAM (target +5–12 %)

- **L1 — Cache cuBLASLt algo for bias-epilogue GEMMs.**
  The TF32/FP16 GEMM fast path already caches descriptors+algo by shape
  (`CudaMatmul.cu:592`, `CudaAmp.cu:589`), but the **bias-epilogue** path (FFN
  gate+up / down, `CudaFeedForward.cu:214`) re-runs `cublasLtMatmulAlgoGetHeuristic`
  every call. Extend the shape-keyed cache to the bias path.
  Gate: ≥3 % mean tok/s (the explicit CONTRIBUTING gate). Risk: low.

- **L2 — Pack/seq sweep for max GPU utilization.**
  4096 tokens/step may under-fill the GPU and over-weight per-step host-SGD +
  launch overhead. Sweep `set_max_packed_columns` (and seq 384/512/768) to find
  the tok/s peak within VRAM. Risk: low (VRAM-bounded); pairs with L4.

- **L3 — Tighten host-SGD overlap.**
  If Step 0 shows exposed `d2h`/`sgd`/`h2d`, batch fused-grad D2H
  (`CudaSbao::commitPinnedD2hBatch`), raise async depth (`kMaxAsyncHostUpdate`,
  currently 4) and/or dedicate copy streams so host SGD hides fully behind
  `bwdGpu`. Gate: shrink exposed copy phases → net tok/s. Risk: medium
  (pipeline correctness; do not stack speculative stream rewrites).

### Phase 2 — recompute policy (largest single lever, target +10–20 %)

- **L4 — Replace Full checkpointing with a budgeted/strided policy.**
  Full ckpt recomputes **all 34 block forwards** during backward (~one extra
  full forward per step). Use the ~3.6 GiB headroom to retain activations for as
  many layers as fit and recompute only the rest (checkpoint *stride*, or
  `Selective` which drops only attention activations). Fewer recomputes = higher
  tok/s at the cost of VRAM.
  Where: `ActivationCheckpointMode` + block loop in
  `CudaLanguageModel.cu:465`, `LanguageModel.hpp`. Gate: tok/s up **and** no OOM
  across warm runs (respect the WDDM VRAM floor notes in
  `CudaLanguageModel.cu:1291`). Risk: medium (VRAM).

- **L5 — Cheaper recompute when VRAM is tight.**
  If L4 headroom is too small, recompute only the FFN (Selective) instead of the
  whole block; attention backward already reuses the stored flash log-sum-exp
  (`CudaFlashAttention`), so its recompute is avoidable. Gate: tok/s vs Full.

### Phase 3 — head/CE and numerics (target +5–10 %)

- **L6 — Fuse the LM head + cross-entropy.**
  Head/CE is forced FP32 over 32k vocab and materializes ~512 MiB logits in
  chunks (`CudaLanguageModel.cu:2343`). Fuse chunked online-softmax with the
  head-gradient pass and/or partition the vocab to cut logits traffic (keep FP32
  accumulation for stability). Gate: `headCe` share drops → net tok/s.

- **L7 — Evaluate bf16 working weights/activations.**
  Everything is FP16 + dynamic loss scaling today (`CudaAmp.cu`); no bf16 path.
  On sm_120 (Blackwell), bf16 tensor cores are strong and remove loss-scaling
  overhead/overflow zeroing. Prototype a bf16 GEMM path with a host/GPU parity
  smoke before trusting throughput. Gate: parity holds **and** ≥3 % tok/s. Risk:
  high (numerics, broad change) — do last.

### Phase 4 — launch overhead / graphs (target +0–5 %)

- **L8 — Fuse small elementwise kernels.**
  Residual-add + RMSNorm-apply + SwiGLU are separate 256-thread launches
  (`CudaOps.cu`, `CudaRMSNorm.cu`, `CudaFeedForward.cu`). If nsys shows the step
  is launch-bound, fuse adjacent elementwise ops. Gate: ≥3 %.
- **L9 — CUDA graphs.** Only viable with checkpointing **off** and fixed shapes
  (`CudaLanguageModel.cu:638`); not applicable while recompute is active at 4B.
  Low priority unless L4 yields a graph-able fixed-shape variant.

## Rough path to +33 %

Compounding a plausible measured mix — L4 partial recompute (~+15 %) × L2 pack
(~+6 %) × L1/L3 copy+algo (~+5 %) × L6 head/CE (~+5 %) ≈ **1.34×** → ~1,300 tok/s.
The real mix depends entirely on the Step 0 attribution; treat these as targets,
not promises.

## Guardrails

- Keep CE/head FP32 accumulation; keep dynamic loss scaling until L7 proves bf16.
- Do not hold the buffer pool (holding it drops 4B to ~236 tok/s —
  `CudaMatmul.cu:161`); `cudaMallocAsync` stays disabled on WDDM
  (`CudaMatmul.cu:74`).
- One change at a time, warm multi-run mean, keep/revert per
  [`CONTRIBUTING.md`](../CONTRIBUTING.md). Re-measure on locked clocks / Linux.

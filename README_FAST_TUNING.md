# Fast tuning patch

This patch keeps the original `scripts/tune_fusion.py` untouched and adds a
hardware-accelerated equivalent, `scripts/tune_fusion_fast.py`.

## What changes

The legacy tuner repeatedly:

- recomputes pairwise JSD for each `(m, agreement)` combination;
- stores six copies of full float32 probabilities per condition;
- recomputes the same `(config, condition)` fusion for every LOFO fold/global run;
- performs the 1000-class fusion in NumPy on CPU.

The fast tuner instead:

1. computes pairwise JSD once per condition;
2. derives all agreement/stability variants from that pass;
3. keeps one probability tensor per condition;
4. evaluates every config/condition exactly once;
5. batches the 1000-class fusion on CUDA;
6. obtains LOFO/global winners by taking means over a tiny accuracy matrix.

It searches the same grid and writes the same `held_out` / `global` JSON
structure as the old tuner. Fast tuning files are named `tuning_fast_...json`
to avoid mixing them with legacy outputs.

## Your RTX 5070

Start conservatively with a config batch of 64:

```powershell
.\.venv\Scripts\python.exe scripts\tune_fusion_fast.py `
  --models resnet18 `
  --cache-dir cache_fixed_fresh `
  --tag cache_fixed_fresh `
  --out results_fast `
  --tune-severities 1 3 5 `
  --device cuda `
  --config-batch 64
```

If Task Manager / `nvidia-smi` shows plenty of free VRAM, try 96 or 128.
The script only holds one condition in VRAM at a time.

Do **not** enable `--tf32` for the first validation run. It is faster, but can
introduce small numerical differences. After verifying the fast tuner chooses
the same configs as the legacy tuner, `--tf32` can be benchmarked separately.

## Full 2x2 run using the existing caches

```powershell
.\.venv\Scripts\python.exe scripts\run_ablation_fast.py `
  --fixed-cache cache_fixed_fresh `
  --geometric-cache cache_geometric_fresh `
  --results results_fast `
  --device cuda `
  --config-batch 64 `
  --force-tuning
```

This does **not** rebuild the 20 GB of caches. It reruns the grid with the fast
backend, then CE fitting/evaluation.

## Validation

The patch includes `test/test_tune_fusion_fast.py`. It checks that:

- one-pass signal computation matches the project's existing C/A/S functions;
- batched PyTorch scoring gives exactly the same accuracies as the legacy CPU
  scoring on synthetic data.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  test\test_tune_fusion_fast.py `
  test\test_analysis.py `
  test\test_ce_fusion.py
```

## Expected speedup

The exact gain depends on disk/CPU/GPU, but your previous logs show roughly
10-11 minutes of repeated signal precompute and ~20-25 minutes of repeated grid
scoring per model/cache. The new structure should cut the signal stage toward
the CE tuner's ~2 minutes and remove the repeated fold scoring entirely. On an
RTX 5070, the remaining grid work should be much faster than the NumPy loop.

Benchmark first rather than assuming a specific multiplier. The output prints
separate `signals` and `grid` timings per condition so the bottleneck is visible.

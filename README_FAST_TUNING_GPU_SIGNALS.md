# Fast tuning v2: CUDA signals + CUDA grid

This patch extends `scripts/tune_fusion_fast.py` so the remaining signal bottleneck
also runs on PyTorch/CUDA.

## What changed

The previous fast tuner already moved the 1,860-config fusion grid to the GPU,
but C/A/S preprocessing still used NumPy on the CPU. On the RTX 5070 benchmark
that reduced a ~36 minute legacy run to ~115 seconds, of which ~95 seconds were
signal computation and only ~4 seconds were grid scoring.

The updated default (`--signal-backend auto`) uses CUDA for both stages when
`--device cuda` is active:

1. copy one cached condition to VRAM once;
2. compute confidence on GPU;
3. compute pairwise base-2 JSD on GPU;
4. derive agreement-all, agreement-to-origin, and stability m=1/2/3 on GPU;
5. standardize signals on GPU;
6. evaluate the full config grid in GPU batches;
7. copy only the small accuracy vector back to CPU.

The JSD implementation computes unique upper-triangle view pairs in bounded
pair batches and mirrors them. This avoids a huge `(N,V,V,C)` temporary.

The old NumPy signal implementation is still available for parity/debugging:

```powershell
.\.venv\Scripts\python.exe scripts\tune_fusion_fast.py `
    --models resnet18 `
    --cache-dir cache_fixed_fresh `
    --out results_fast `
    --device cuda `
    --signal-backend numpy
```

## Recommended RTX 5070 command

Start with the safe defaults:

```powershell
.\.venv\Scripts\python.exe scripts\tune_fusion_fast.py `
    --models resnet18 `
    --cache-dir cache_fixed_fresh `
    --tag cache_fixed_fresh `
    --out results_fast_gpu_signals `
    --tune-severities 1 3 5 `
    --device cuda `
    --config-batch 64 `
    --signal-chunk 256 `
    --signal-pair-batch 32
```

`--signal-backend auto` is the default, so CUDA signals are selected automatically.
To make it explicit, add `--signal-backend torch`.

For an 8-view spectral profile there are only 28 unique view pairs, so
`--signal-pair-batch 32` computes all pairs at once. For a 15-view combined
profile there are 105 pairs and they are processed in four pair batches.

If VRAM headroom is large, `--signal-chunk 512` is worth benchmarking. It may
reduce kernel-launch overhead, but 256 is the conservative default and is also
safer for 15-view combined profiles.

## Correctness

The tests compare the Torch JSD/signals with the canonical NumPy definitions,
including exact-zero probabilities, and verify that the resulting grid
accuracies are unchanged on synthetic inputs.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q test\test_analysis.py test\test_tune_fusion_fast.py
```

## Expected bottleneck after this patch

The cache `.npz` load/decompression and host-to-device transfer may become a
larger fraction of total runtime. At that point further GPU optimization has
diminishing returns; the scientific grid itself is already only a few seconds.

# 2x2 ablation patch

This patch implements:

1. `geometric` spectral ladder with the same endpoints and K=7 as `fixed`.
2. Non-negative cross-entropy fitting of alpha/beta/gamma with tau=1.
3. NLL reporting in the existing evaluation pipeline.
4. A 2x2 orchestration script for fixed/geometric x grid/CE.
5. Unit tests for the CE objective and gradient.
6. The retained-energy correction `power * mask**2` for the old energy ladder.

## Changed/new files

- `src/cache.py`
- `src/metrics.py`
- `src/ce_fusion.py` (new)
- `scripts/cache_profiles.py`
- `scripts/evaluate.py`
- `scripts/tune_fusion_ce.py` (new)
- `scripts/run_ablation.py` (new)
- `test/test_ce_fusion.py` (new)

## Run order

Your existing `cache_fixed/` remains valid and should NOT be regenerated.

### 1. Fit CE on the existing fixed cache first

```powershell
.\.venv\Scripts\python.exe scripts\tune_fusion_ce.py --cache-dir cache_fixed --grid-tuning auto
```

Then evaluate the new CE tuning file explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --cache-dir cache_fixed --tuning results\tuning_ce_resnet18_cache_fixed_<timestamp>.json --models resnet18 --tag fixed_ce
```

(and the analogous DeiT command).

### 2. Build the geometric cache once

```powershell
.\.venv\Scripts\python.exe scripts\cache_profiles.py --ladder geometric --cache-dir cache_geometric
```

### 3. Full 2x2 ablation

After both caches exist:

```powershell
.\.venv\Scripts\python.exe scripts\run_ablation.py
```

If you want the orchestrator to build a missing geometric cache itself, use:

```powershell
.\.venv\Scripts\python.exe scripts\run_ablation.py --build-geometric-cache
```

The script writes:

- `results/ablation_2x2_<timestamp>.csv`
- `results/ablation_2x2_<timestamp>.json`

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q test\test_analysis.py test\test_ce_fusion.py
```

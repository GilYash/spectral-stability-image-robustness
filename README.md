# Spectral Stability Profiles for Robust Test-Time Image Classification

Modern Computer Vision final project. Tomer Chaiat, Omer Yahav, Hadar Etzion,
Gil Yashayev, Saar Misk.

A per-image **spectral stability profile** is a frozen classifier's ordered
response to progressively low-pass filtered versions of a test image. We ask
whether that profile can be used to adaptively combine frequency views at
inference time and improve corruption robustness — with no retraining, no
corruption labels, no source data and no parameter updates.

## Core idea, and why the code is shaped this way

For a test image `x` we build `K+1` views `x_k = F^-1(F(x) * M_k)` where `M_0`
is the identity and the rest are progressively stronger smooth radial low-pass
masks. Each view goes through the same frozen classifier, giving the profile
`S(x) = (p_0, ..., p_K)`. Everything after that — the reliability signals
`C_k, A_k, S_k`, the fusion weights, every baseline and every ablation — is a
**pure function of `S(x)`**.

So the pipeline is split in two:

1. **GPU, once.** `scripts/02_cache_profiles.py` computes `S(x)` for every
   (model, corruption, severity) and writes it to `cache/`.
2. **CPU, forever.** Every experiment afterwards is numpy over cached arrays.
   Hyper-parameter sweeps and ablations take seconds, not GPU hours.

Do not add a code path that runs the model during analysis.

## Layout

```
src/filters.py    radial low-pass mask bank + spectral energy analysis
src/spatial.py    spatial TTA view bank (the equal-view-count control)
src/data.py       stratified ImageNet subset, ImageNet-C style corruption
src/models.py     frozen ResNet-18 / DeiT-Tiny
src/cache.py      profile precomputation, sweep driver, loading
src/signals.py    C_k, A_k, S_k, softmax fusion    (pure numpy)
src/sweep.py      digest cache + fast grid search  (pure numpy)
src/baselines.py  baselines + oracle bounds        (pure numpy)
src/metrics.py    accuracy, ECE, negative/positive transfer
src/splits.py     corruption families, leave-one-family-out, image splits
src/plotstyle.py  shared figure style, fixed categorical hues
scripts/          runnable pipeline, numbered in execution order
report/           compiling LaTeX skeleton + refs.bib; tables auto-generated
run_all.sh        one-shot driver for the cluster
test_numpy_paths.py  offline checks, no GPU or data needed
make_fake_cache.py   synthetic cache to exercise the pipeline without ImageNet
```

## Pipeline

| Step | Needs | Cost |
|---|---|---|
| `00_build_subset.py` | ImageNet val | seconds |
| `01_sanity_filters.py` | one image | seconds |
| `02_cache_profiles.py` | **GPU + many CPU cores** | hours, once |
| `03_oracle_gonogo.py` | cache | seconds |
| `04_tune.py` | cache | ~5 min per model |
| `05_evaluate.py` | cache + configs | minutes |
| `06_ablations.py` | cache | minutes |
| `07_figures.py` / `08_tables.py` | results | seconds |

Or just `./run_all.sh /path/to/imagenet/val`, which stops at the go/no-go and
asks before spending the night on the full sweep.

## Quickstart (cluster)

```bash
pip install -r requirements.txt
python test_numpy_paths.py                              # should print ALL CHECKS PASSED

python scripts/00_build_subset.py --val-root /path/to/imagenet/val
python scripts/01_sanity_filters.py --image <any val image>   # LOOK at the output
python scripts/02_cache_profiles.py --models resnet18 \
    --corruptions clean gaussian_noise defocus_blur contrast --severities 5
python scripts/03_oracle_gonogo.py --model resnet18
```

To exercise the analysis half without ImageNet or a GPU:

```bash
python make_fake_cache.py && python scripts/04_tune.py --models resnet18
```

## Setup

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS SSL certificate issue

On some macOS Python installations, downloading pretrained model weights may
fail with `CERTIFICATE_VERIFY_FAILED`.

For a Python.org installation, run the certificate installer corresponding to
your Python version. For example, for Python 3.14:

```bash
open "/Applications/Python 3.14/Install Certificates.command"
```

Then retry the model download.

## Read the go/no-go before building anything else

`03_oracle_gonogo.py` prints, per corruption:

```
original            0.xxxx
oracle (any view)   0.xxxx
HEADROOM            +0.xxxx
```

Headroom is the ceiling for *any* per-image weighting scheme over this view
family. If it is small, no scoring function fixes it and the views themselves
need to change (different cutoffs, band-pass instead of low-pass, more levels).
Finding that out in week 1 is cheap; finding it out in week 4 is not.

## Conventions that are easy to get wrong

- **Class indices stay 0..999.** The 100-class subset keeps original ImageNet
  indices so pretrained heads line up. Never re-index to 0..99.
- **Filter in `[0,1]` image space, before normalisation.** Resize(256) →
  centre-crop(224) → corrupt → to float → filter → normalise.
- **Corrupt after cropping.** ImageNet-C was generated on 224×224 crops.
- **Identical geometry for both architectures**, so ResNet and DeiT see exactly
  the same corrupted pixels; only the normalisation constants differ.
- **Signals are z-scored per image before fusion** (`standardize=True`).
  `C_k ∈ [0,1]` and, since Jensen–Shannon divergence is computed with `log2`,
  `A_k, S_k ∈ [-1, 0]`. Without standardization, the scales and distributions
  of the three signals are still different, so α, β, γ are not directly
  comparable and τ is harder to interpret. This is a design choice — ablate it.
- **Tuning is leave-one-corruption-family-out AND on a disjoint image split.**
  Report held-out numbers only.

## Cost and storage

**Corruption generation, not the GPU, is the bottleneck.** `glass_blur` alone
costs more CPU than every other corruption combined, so the sweep driver
generates each condition's corrupted pixels once with a process pool and shares
them across every model and view kind. Use `--skip-slow` on the first pass.

One condition is `(N, K+1, 1000)` float16 ≈ 60 MB at N=5000, K=5. The full
sweep (2 models × 2 view kinds × (1 clean + 15 corruptions × 5 severities)) is
≈ 18 GB — check your quota before launching. Halve it with `--n-per-class 25`;
the method comparisons are paired on the same images, so their error bars are
much tighter than the raw per-condition accuracies suggest.

Cache, results and figures are gitignored.
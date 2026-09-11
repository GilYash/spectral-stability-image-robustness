# Spectral Stability Profiles for Robust Test-Time Image Classification

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![Models](https://img.shields.io/badge/models-ResNet--18%20%7C%20DeiT--Tiny-44cc11.svg)
![Benchmark](https://img.shields.io/badge/benchmark-ImageNet--C-f39c12.svg)

> Modern Computer Vision final project - Tomer Chaiat, Omer Yahav, Hadar Etzion, Gil Yashayev, Saar Misk

## Overview

When a photo is corrupted - sensor noise, motion blur, fog, low contrast - a
frozen classifier often gets it wrong.  
This project asks whether it can recover
at test time using nothing but its own reactions to that one image: no
retraining, no labels, no access to the corruption, no weight updates.

The idea: take the test image, blur it a little, blur it more, blur it a lot,
and run every version through the same classifier. That gives a short
trajectory of predictions - a **spectral stability profile**. Where the
classifier is confident and its answer holds steady as the blur changes, trust
that view; where it flip-flops, don't.  
Then average the views with those trust weights.

### 🗝️ Key Ideas

- **Frozen classifier** - no retraining, no corruption labels, no source data, no parameter updates.
- **Frequency views** - the image low-pass filtered at a ladder of strengths (two ladders are compared: per-image energy-adaptive, and fixed absolute cutoffs - see Results below).
- **Three trust signals** - confidence, agreement across views, and stability along the ladder, combined into softmax fusion weights.
- **Honest evaluation** - coefficients tuned leave-one-corruption-family-out, on a disjoint half of the images.
- **Two architectures** - ResNet-18 and DeiT-Tiny, fed identical pixels, to test whether a CNN and a ViT behave differently.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A GPU for the one-time caching step (everything after runs on CPU)
- ~15 GB free disk for the data and the profile cache

### Installation & Setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

python test_models.py                        # pull the pretrained weights
PYTHONPATH=. python test/test_analysis.py    # should print "all passed"
```

Then build the dataset - a 100-class slice of ImageNet-C plus the matching
clean validation images. Full instructions and checks are in
[`data/README.md`](data/README.md); the short version:

```bash
python scripts/build_subset.py          # choose the 100 classes
python scripts/fetch_clean_images.py    # clean images
python scripts/fetch_imagenet_c.py      # corrupted images (safe to stop and re-run)
```

## 💻 Usage

The pipeline is split so the GPU work happens once and everything after is fast
NumPy you can re-run while changing parameters.

### The pipeline

```bash
python scripts/cache_profiles.py            # GPU, once: every profile -> cache/  (~40 min/model, ~10 GB)
python scripts/tune_fusion.py               # grid-search the fusion coefficients, leave-one-family-out
python scripts/evaluate.py --tuning auto    # score every condition -> results/eval_<model>_<time>.csv
```

### Try one setting by hand

```bash
python scripts/evaluate.py --alpha 1 --beta 0.5 --gamma 0.5 --tau 0.1
```

### Inspect a single image

```bash
python scripts/profile_single_image.py --image data/imagenet-c/gaussian_noise/3/<wnid>/<img>.JPEG
python scripts/show_filter_views.py    --image <same image>
python scripts/verify_filter_energy.py --image <same image>
```

## 📊 Results

100-class ImageNet-C, held-out image half, leave-one-family-out configs. Mean
over 15 corruptions x 5 severities. Two view families are compared: **energy**
(the original design, cutoffs solved per image to keep a target fraction of
that image's spectral energy) and **fixed** (same absolute cutoffs for every
image - the full reason is in Where to Push Next below):

| metric | ResNet-18 energy | ResNet-18 fixed | DeiT-Tiny energy | DeiT-Tiny fixed |
|---|:--:|:--:|:--:|:--:|
| original view | 32.6% | 32.6% | 44.0% | 44.0% |
| fused (tuned) | 31.6% | **33.5%** | 40.8% | **44.0%** |
| fused - original | -0.9 pts | **+0.9 pts** | -3.2 pts | **+0.0 pts** |
| per-image oracle | 39.3% | 41.0% | 50.6% | 52.4% |
| clean (orig / fused) | 68.9 / 64.7 | 68.9 / 67.7 | 70.1 / 66.1 | 70.1 / 68.7 |
| mean negative transfer | 13.7% | 8.5% | 14.5% | 8.0% |

- **The energy ladder does not work; fixed cutoffs do.** Under `energy`, tuning
  collapses to confidence-only (beta = gamma = 0, near-hard view selection) on
  both models - agreement and stability carry no signal. Under `fixed`, every
  fold uses stability too (and DeiT uses agreement), fusion actually beats the
  baseline for ResNet-18, and negative transfer is roughly halved on both
  models. Why: the energy ladder's "mild" cutoffs are
  only mild for high-frequency corruptions.
- **Corruption-dependent, as predicted.** Under `fixed`, fusion clearly wins on
  the three noise types and pixelate (+6 to +9 points), and is close to flat on
  most blur corruptions; weather and contrast/jpeg/elastic remain small net
  losses.
- **Architecture-dependent, as predicted.** ResNet-18 gets a real net win from
  low-pass views; DeiT-Tiny only reaches parity. Consistent with DeiT's
  self-attention already acting like a low-pass filter - it starts more robust
  (44.0% vs 32.6% mean corrupted) and has less to gain from an external one.

## 🏗️ Architecture

### How it works

For a test image `x` we build `K + 1` views:

```
x_0 = x                          the original, untouched
x_k = F^-1( F(x) · M_k )          k = 1 ... K
```

`F` is the 2-D DFT and each `M_k` is a smooth Gaussian low-pass mask. The masks
get progressively stronger, and the cutoff for each view is solved **per image**
so the view keeps a target fraction of that image's spectral energy (default
ladder: keep 99%, 98%, ... 93%).

Every view goes through the **same frozen classifier**, giving the profile
`S(x) = (p_0, ... , p_K)`. From it we read three trust signals per view:

| signal | what it measures | formula |
|---|---|---|
| `C_k` confidence | how peaked the prediction is | `max_c p_k(c)` |
| `A_k` agreement | how close this view is to all the others | `-(1/K) Σ_{j≠k} JSD(p_k, p_j)` |
| `S_k` stability | how close this view is to its ladder neighbours | `-mean_{j∈N(k)} JSD(p_k, p_j)` |

`JSD` is the base-2 Jensen-Shannon divergence, so `C_k ∈ [0, 1]` and
`A_k, S_k ∈ [-1, 0]`. The three combine into a score and softmax weights, and
the views are averaged:

```
r_k = α·C_k + β·A_k + γ·S_k
w   = softmax(r / τ)
p*  = Σ_k w_k · p_k
```

`p*` is judged against the original view alone, an equal-weight average, a
confidence-only weighting, and a per-image oracle (right if *any* view is - the
ceiling for any weighting scheme).

That per-image energy-adaptive cutoff is one of two ladders implemented -
`src/cache.py` also has `spectral_views_fixed`, the same absolute cutoffs for
every image, which the results above show working a lot better. See
Where to Push Next below for why.

### Project Structure

```
src/
  filters.py           low-pass mask bank + the per-image energy ladder
  analysis.py          the profile maths: C_k, A_k, S_k and softmax fusion
  metrics.py           accuracy, ECE, oracle bound, negative / positive transfer
  splits.py            corruption families, leave-one-family-out, disjoint image halves
  cache.py             build and load the profile cache
  models.py            frozen ResNet-18 and DeiT-Tiny
  imagenet_labels.py   image path -> true ImageNet index

scripts/
  build_subset.py           pick the fixed 100-class subset          |
  fetch_clean_images.py     clean ImageNet-val for those classes     |  data - see data/README.md
  fetch_imagenet_c.py       download ImageNet-C for those classes    |
  cache_profiles.py         GPU, once: compute every profile -> cache/
  tune_fusion.py            grid-search alpha, beta, gamma, tau
  evaluate.py               score every condition, write one results table
  show_filter_views.py      \
  verify_filter_energy.py    >  look at one image
  profile_single_image.py   /

test/test_analysis.py       unit tests for the profile maths (no GPU or data needed)
```

## ⚠️ Gotchas

- **Class indices stay 0-999.** The subset keeps original ImageNet indices so
  the pretrained heads line up. `src/imagenet_labels.py` reads
  `data/subset_classes.json`; never re-index to 0-99.
- **Filter in `[0, 1]` space, before normalisation.** Order: resize(256) ->
  centre-crop(224) -> to float -> low-pass -> ImageNet-normalise. ImageNet-C
  images already carry the corruption; we do not re-corrupt.
- **Both architectures see identical pixels** - same geometry, only the
  normalisation constants differ.
- **Standardise the signals before combining.** Confidence is in `[0, 1]`,
  agreement and stability in `[-1, 0]`, so without the per-image z-score
  (`standardize`, on by default) alpha, beta, gamma are not comparable. Whether
  to standardise is itself part of the tuning grid.
- **Tuning never sees the family it is scored on**, and tunes on a disjoint
  image half.

## 🔬 Where to Push Next

**Diagnosis that led to the fixed ladder.** Splitting per-view accuracy by
corruption family showed the energy ladder's first, "mildest" rung already
crushes blur/weather/digital while staying gentle for noise. The reason: its
cutoff is solved per image to keep a target *fraction of that image's own
spectral energy*, and a blur- or fog-corrupted image's spectrum is already
concentrated near DC (the corruption removed the high frequencies itself), so
keeping 99% of its energy takes a narrow, aggressive mask - solved cutoff 0.12
for defocus_blur vs 0.66 for gaussian_noise at the *same* nominal target.
`spectral_views_fixed` in `src/cache.py` uses the same absolute cutoffs for
every image instead, which is the whole effect in the results table above.
Confirmed this was the real bottleneck, not the fusion coefficients: tuning
*per corruption* (maximum possible specialisation) gave the same numbers as
family-level tuning - no amount of re-weighting fixes a bad view family.

Build and evaluate the fixed ladder yourself:

```bash
python scripts/cache_profiles.py --ladder fixed --cache-dir cache_fixed
python scripts/tune_fusion.py --cache-dir cache_fixed
python scripts/evaluate.py --cache-dir cache_fixed --tuning auto
```

Still open - weather and contrast/jpeg/elastic remain small net losses even
under the fixed ladder, and DeiT-Tiny only reaches parity, not a win:

- a milder or differently-spaced `FIXED_CUTOFFS`, calibrated per family;
- **band-pass** views instead of low-pass, so mid frequencies survive;
- more views, or views chosen per image;
- why agreement stays at zero weight for ResNet-18 even now - nested low-passes
  are correlated by construction and can form a self-agreeing cluster.

Everything after `cache_profiles.py` runs in seconds, so this is cheap to
explore.

## 🐛 Troubleshooting

- **`no results/tuning_<model>_*.json found`** - run `scripts/tune_fusion.py`
  before `scripts/evaluate.py --tuning auto`.
- **`evaluate.py` / `tune_fusion.py` skip everything** - they read `cache/`; run
  `scripts/cache_profiles.py` first.
- **ImageNet-C download is slow or drops mid-way** - expected, the source
  server throttles. Re-run `scripts/fetch_imagenet_c.py`; it resumes.
- **macOS `CERTIFICATE_VERIFY_FAILED`** when pulling model weights - run the
  Python cert installer, e.g.
  `open "/Applications/Python 3.12/Install Certificates.command"`.

## 📎 Reference

ImageNet-C: Hendrycks and Dietterich, *Benchmarking Neural Network Robustness to
Common Corruptions and Perturbations*, ICLR 2019
([Zenodo 2235448](https://zenodo.org/records/2235448)).


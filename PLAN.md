# Practical phase plan — front-loaded

**Hard deadline: 30 September 2026.** Report (2–3 pages, LaTeX), public GitHub
repo, 5-minute talk.

Everything that can be done without results is already done: the pipeline, the
analysis code, the figure and table generators, and a compiling report skeleton.
Week 1 runs the experiments and writes the paper. Weeks 2–4 are fixes only.

## Why one week is enough

The GPU work is a few hours. Everything after it is numpy over cached arrays, so
the tune → evaluate → ablate → figures → tables chain runs in minutes and can be
re-run as many times as you like. The only genuinely serial thing is the cache.

**The real bottleneck is CPU, not GPU.** Corruption generation dominates —
`glass_blur` alone costs more than every other corruption combined. It is
process-parallel and generated once per condition and shared across both models
and both view kinds. Use `--skip-slow` for the first full pass and add
`glass_blur` afterwards if time allows.

---

## Day 1 — Tue 2 Sep · get the cache running, in parallel with everything else

**Person 2 (cluster), serial and blocking:**

- [ ] `pip install -r requirements.txt`, `python test_numpy_paths.py`
- [ ] `python scripts/00_build_subset.py --val-root <val>`
- [ ] `python scripts/01_sanity_filters.py --image <a val image>` — **look at it**
- [ ] Go/no-go cache: `--models resnet18 --corruptions clean gaussian_noise
      defocus_blur contrast --severities 5` (~20 min)
- [ ] `python scripts/03_oracle_gonogo.py --model resnet18 | tee results/gonogo.txt`
- [ ] **Decision gate below.** Then launch the full sweep overnight:
      `python scripts/02_cache_profiles.py --view-kinds lowpass spatial --skip-slow`

**Everyone else, same day, no cache needed:**

- [ ] Person 1: `python scripts/07_figures.py --only spectral` — this is the H1
      figure and needs no model at all, only images. Do it today.
- [ ] Persons 3 & 4: write `report/main.tex` intro, related work and method.
      All of it is known now; none of it depends on results.
- [ ] Person 5: talk skeleton — 5 slides, the H1 figure is slide 2.

### Decision gate (day 1, before the overnight run)

`results/gonogo.txt` prints `HEADROOM` = oracle-over-views minus original. It is
the ceiling for *any* per-image weighting over this view family.

- Healthy headroom → launch the full sweep, continue as planned.
- Thin headroom → **do not** spend the night on it. Change the view family
  first: widen the cutoff spread, add levels, or try band-pass instead of pure
  low-pass (`DEFAULT_CUTOFFS` in `src/filters.py` is the only thing to edit).
  Re-run the 20-minute go/no-go on the new bank.

## Day 2 — Wed 3 Sep · all results, in one afternoon

- [ ] Cache finished overnight; sanity-check sizes and counts
- [ ] `python scripts/04_tune.py` (~5 min/model)
- [ ] `python scripts/05_evaluate.py` → `results/results.csv`
- [ ] `python scripts/06_ablations.py` → `results/ablations.csv`
- [ ] `python scripts/07_figures.py` and `python scripts/08_tables.py`
- [ ] Read the numbers together. Decide what the paper's claim actually is.

Anything wrong here is a numpy re-run, not a GPU re-run. That is the point of
the two-phase design.

## Day 3 — Thu 4 Sep · fix, re-run, and lock the story

- [ ] Whatever day 2 exposed: re-tune, re-evaluate, regenerate
- [ ] Add `glass_blur` to the cache if the first pass skipped it
- [ ] Confirm H1 and H2 against the figures — and write down honestly if either
      is not supported
- [ ] Freeze results. No more experiments after today.

## Day 4 — Fri 5 Sep · write

- [ ] Results and discussion sections; tables are already generated
- [ ] Abstract last, with the headline number and its sign
- [ ] Full read-through by someone who did not write it

## Day 5 — Sat 6 Sep · talk and repo

- [ ] Slides built from the paper figures; **rehearse to 5 minutes**
- [ ] Repo cleaned, README reproducible from scratch, made public
- [ ] `run_all.sh` verified on a fresh clone

## Days 6–7 — Sun 7 / Mon 8 Sep · buffer

Deliberately empty. Something will slip into it.

---

## Weeks 2–4 (9–30 Sep) — fixes only

No new experiments planned. Keep these in reserve:

- Re-run with more images per class if reviewers of your own draft want tighter
  error bars (`--n-per-class`)
- Add the band-pass view family as a second experiment *if* low-pass headroom
  was thin and you want a positive result to report
- Polish figures, tighten the report to the page limit
- Re-rehearse the talk once the date is announced

Check in once a week. If nothing is broken, do nothing.

## Work split

| Who | Day 1 | Day 2–3 | Day 4–5 |
|---|---|---|---|
| 1 | H1 spectral figure (no GPU needed) | figures, verify H1/H2 | figure polish |
| 2 | cluster, subset, go/no-go, overnight cache | tune + evaluate | repo public, run_all check |
| 3 | report intro + related work | ablations | results section |
| 4 | report method section | tables, calibration numbers | discussion + abstract |
| 5 | talk skeleton | read results critically | build and rehearse talk |

## Honesty commitments from the proposal

Non-negotiable in the report even if unflattering:

- Negative-transfer rate reported alongside every gain
- Oracle upper bound reported, so the reader sees the headroom
- α, β, γ, τ tuned on held-out corruption families, then frozen
- Results broken out per corruption, severity, architecture, filter strength
- H1 predicts corruption-dependent gains — that dependence is the *predicted
  outcome*, not a failure mode

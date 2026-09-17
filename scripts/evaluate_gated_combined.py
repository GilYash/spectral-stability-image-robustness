"""Evaluate a cheap spectral gate for the 15-view combined profile.

Motivation
----------
The full combined method always pays for 15 classifier forward passes:

    1 original + 7 spectral low-pass views + 7 extra spatial-TTA views.

Plain spatial TTA needs only 8 passes (original + 7 crop/flip variants), and
our previous results show that spectral views mainly help images with strong
high-frequency corruption.  This script tests a conditional policy:

    low high-frequency energy  -> use spatial TTA only        (8 forwards)
    high high-frequency energy -> use full combined fusion   (15 forwards)

The gate itself uses only the input image FFT, so it does not require another
classifier forward pass.  The script tunes the FFT radius/threshold on the
same tune half and leave-one-corruption-family-out protocol as tune_fusion.py,
then reports accuracy on the untouched held-out test half.

Crucially, this is an OFFLINE simulation over existing caches: no model
inference is rerun.  It tells us whether conditional spectral inference is
worth implementing in the online pipeline.

Prerequisites
-------------
  * cache_spatial/   : scripts/cache_spatial_profiles.py
  * cache_combined/  : scripts/build_combined_cache.py
  * results/tuning_<model>_cache_combined_*.json from tune_fusion.py

Example
-------
    python scripts/evaluate_gated_combined.py --tuning auto

Try several compute budgets at once:
    python scripts/evaluate_gated_combined.py --tuning auto \
        --budgets 8 9 10 11 12 13 14 15

A budget is the maximum *average* number of classifier forwards on the tuning
split.  With 8-pass spatial TTA and a 15-pass combined profile, budget B
allows at most (B - 8) / 7 of images to take the expensive branch.
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import cache
from src.metrics import FusionConfig, fused_probs, precompute
from src.splits import FAMILIES, CORRUPTION_FAMILY, leave_one_family_out, tune_test_masks


SPATIAL_FORWARDS = 8
FULL_COMBINED_FORWARDS = 15
FULL_LADDER_LEN = 8  # original + seven spectral views in cache_combined
SPECTRAL_CUTOFFS = {
    1: 0.90,
    2: 0.75,
    3: 0.60,
    4: 0.45,
    5: 0.30,
    6: 0.20,
    7: 0.12,
}


def latest_tuning(model_name, tag="cache_combined", results_dir="results"):
    """Newest tuning JSON matching the combined cache for ``model_name``."""
    pattern = str(Path(results_dir) / f"tuning_{model_name}_{tag}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(
            f"No combined tuning file found for {model_name}. Expected {pattern}.\n"
            "Run: python scripts/tune_fusion.py --cache-dir cache_combined "
            "--ladder-len 8"
        )
    return Path(files[-1])


def load_tuning(path):
    """Read a tune_fusion.py result file."""
    return json.loads(Path(path).read_text())


def fold_config(tuning, held_out_family):
    """FusionConfig tuned without ``held_out_family``."""
    try:
        return FusionConfig.from_dict(tuning["held_out"][held_out_family]["config"])
    except KeyError as exc:
        raise SystemExit(
            f"Tuning file has no held-out config for {held_out_family!r}"
        ) from exc


def _load_one_image(path):
    """CPU image load using the exact 224x224 geometry used by the cache."""
    return cache.load_image(path)


def high_frequency_ratios(paths, radii, *, device, batch_size=128, workers=8):
    """Compute high-frequency spectral-energy ratios for a list of images.

    For radius r, the gate score is

        sum_{freq_radius >= r} |FFT(x)|^2 / sum_all |FFT(x)|^2.

    Frequencies use the same [-1, 1] radial coordinate convention as the
    project's Gaussian low-pass masks.  Power is summed across RGB channels.

    Returns:
        (N, len(radii)) float32 array.
    """
    radii = np.asarray(radii, dtype=np.float32)
    out = np.empty((len(paths), len(radii)), dtype=np.float32)
    if not len(paths):
        return out

    # All project inputs are 224x224 after cache.load_image().
    h = w = 224
    y = torch.linspace(-1.0, 1.0, h, device=device)
    x = torch.linspace(-1.0, 1.0, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rr = torch.sqrt(xx.square() + yy.square())
    masks = torch.stack([(rr >= float(r)).to(torch.float32) for r in radii], dim=0)

    # Threaded decoding keeps the GPU fed without Windows multiprocessing
    # complications.  workers=0 falls back to a simple serial loader.
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 0 else None
    try:
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            if pool is None:
                imgs = [_load_one_image(p) for p in batch_paths]
            else:
                imgs = list(pool.map(_load_one_image, batch_paths))
            images = torch.stack(imgs, dim=0).to(device, non_blocking=True)

            spec = torch.fft.fftshift(
                torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1)
            )
            power = spec.abs().square().sum(dim=1)  # (B,H,W), RGB summed
            total = power.sum(dim=(-2, -1)).clamp_min(1e-12)  # (B,)
            # (B,R): one matrix multiply instead of one reduction per radius.
            numer = torch.einsum("bhw,rhw->br", power, masks)
            ratios = numer / total[:, None]
            out[start:start + len(batch_paths)] = ratios.detach().cpu().numpy()
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return out


def gate_cache_path(root, corruption, severity):
    sev = 0 if corruption == "clean" else severity
    return Path(root) / corruption / f"{sev}.npz"


def load_or_compute_gate_scores(paths, corruption, severity, radii, *, gate_cache,
                                device, batch_size, workers):
    """Load FFT gate features from disk, or compute and cache them once."""
    out = gate_cache_path(gate_cache, corruption, severity)
    radii_arr = np.asarray(radii, dtype=np.float32)
    paths_arr = np.asarray([str(p) for p in paths])

    if out.exists():
        d = np.load(out, allow_pickle=False)
        if (
            "scores" in d.files
            and "radii" in d.files
            and np.array_equal(d["radii"].astype(np.float32), radii_arr)
            and np.array_equal(d["paths"], paths_arr)
        ):
            return d["scores"].astype(np.float32)

    t0 = time.time()
    scores = high_frequency_ratios(
        list(paths_arr), radii_arr, device=device,
        batch_size=batch_size, workers=workers,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, paths=paths_arr, radii=radii_arr, scores=scores)
    print(
        f"    gate features {corruption}/{severity}: {len(paths):5d} images "
        f"in {time.time() - t0:.1f}s -> {out}"
    )
    return scores


def _aligned_spatial(combined, spatial):
    """Return spatial probabilities reordered to the combined cache path order."""
    cp = list(combined["paths"])
    sp = list(spatial["paths"])
    if cp == sp:
        return spatial["probs"]
    pos = {p: i for i, p in enumerate(sp)}
    if set(cp) != set(sp):
        raise ValueError("combined and spatial caches contain different image sets")
    return spatial["probs"][[pos[p] for p in cp]]


def condition_predictions(model_name, corruption, severity, cfg, mask, *,
                          combined_dir, spatial_dir, spectral_views):
    """Predictions/labels/paths for one condition and one image-half mask.

    Returns full-combined fused predictions and the 8-view equal-average spatial
    TTA predictions.  This is the only expensive NumPy/JSD part of the gate
    experiment; classifier inference is never rerun.
    """
    cache.CACHE_ROOT = Path(combined_dir)
    comb = cache.load_condition(model_name, corruption, severity)

    cache.CACHE_ROOT = Path(spatial_dir)
    spat = cache.load_spatial_condition(model_name, corruption, severity)
    spat_probs = _aligned_spatial(comb, spat)

    # cache_combined layout: [orig, spectral 1..7, spatial 1..7].
    # Keep the original, the requested spectral subset, and every spatial view.
    keep = [0] + list(spectral_views) + list(range(FULL_LADDER_LEN, FULL_COMBINED_FORWARDS))
    comb_probs = comb["probs"][mask][:, keep]
    labels = comb["labels"][mask]
    paths = comb["paths"][mask]

    pre = precompute(
        comb_probs,
        m=cfg.m,
        agreement=cfg.agreement,
        ladder_len=1 + len(spectral_views),
    )
    combined_pred = fused_probs(pre, cfg).argmax(axis=-1)
    spatial_pred = spat_probs[mask].astype(np.float32).mean(axis=1).argmax(axis=-1)

    return combined_pred, spatial_pred, labels, paths


def candidate_thresholds(scores, n=101):
    """Quantile-spaced thresholds plus explicit never/always-gate endpoints."""
    if len(scores) == 0:
        return np.array([np.inf], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n)
    vals = np.unique(np.quantile(scores, qs))
    return np.concatenate(([np.inf], vals[::-1], [-np.inf]))


def threshold_curve(conditions, radius_col, extra_spectral_forwards, n_thresholds=101):
    """Accuracy/compute curve for one FFT radius on tuning conditions.

    Each condition is a dict with ``gate``, ``combined_pred``, ``spatial_pred``,
    and ``labels``.  Accuracy is averaged per condition, matching the project's
    mCA-style tuning rather than pooling all samples into one giant condition.
    """
    all_scores = np.concatenate([c["gate"][:, radius_col] for c in conditions])
    thresholds = candidate_thresholds(all_scores, n=n_thresholds)
    total_n = sum(len(c["labels"]) for c in conditions)

    rows = []
    for t in thresholds:
        accs = []
        gated_n = 0
        for c in conditions:
            gate = c["gate"][:, radius_col] >= t
            pred = np.where(gate, c["combined_pred"], c["spatial_pred"])
            accs.append(np.mean(pred == c["labels"]))
            gated_n += int(gate.sum())
        gate_rate = gated_n / total_n if total_n else 0.0
        rows.append({
            "threshold": float(t),
            "tune_acc": float(np.mean(accs)),
            "gate_rate": float(gate_rate),
            "avg_forwards": float(SPATIAL_FORWARDS + extra_spectral_forwards * gate_rate),
        })
    return rows


def choose_policy(curves_by_radius, budgets):
    """Best tune-accuracy policy under each average-forward budget.

    Ties prefer fewer forwards, then the larger threshold (more conservative
    gating).  Returns budget -> policy dict.
    """
    out = {}
    for budget in budgets:
        feasible = []
        for radius, rows in curves_by_radius.items():
            for row in rows:
                if row["avg_forwards"] <= budget + 1e-9:
                    feasible.append({"radius": float(radius), **row})
        if not feasible:
            raise ValueError(
                f"No feasible policy for budget {budget}; minimum is {SPATIAL_FORWARDS}"
            )
        feasible.sort(
            key=lambda r: (r["tune_acc"], -r["avg_forwards"], r["threshold"]),
            reverse=True,
        )
        out[float(budget)] = feasible[0]
    return out


def evaluate_policy(conditions, radius_col, threshold, extra_spectral_forwards):
    """Evaluate one fixed gate policy over held-out test conditions."""
    rows = []
    for c in conditions:
        gate = c["gate"][:, radius_col] >= threshold
        gated_pred = np.where(gate, c["combined_pred"], c["spatial_pred"])
        n = len(c["labels"])
        gate_rate = float(gate.mean()) if n else 0.0
        rows.append({
            "corruption": c["corruption"],
            "severity": c["severity"],
            "n": n,
            "spatial_acc": float(np.mean(c["spatial_pred"] == c["labels"])),
            "combined_acc": float(np.mean(c["combined_pred"] == c["labels"])),
            "gated_acc": float(np.mean(gated_pred == c["labels"])),
            "gate_rate": gate_rate,
            "avg_forwards": float(SPATIAL_FORWARDS + extra_spectral_forwards * gate_rate),
        })
    return rows


def mean_field(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def weighted_gate_rate(rows):
    n = sum(r["n"] for r in rows)
    return (sum(r["gate_rate"] * r["n"] for r in rows) / n) if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"],
                    choices=["resnet18", "deit"])
    ap.add_argument("--combined-dir", default="cache_combined")
    ap.add_argument(
        "--spectral-views", nargs="+", type=int, default=list(range(1, 8)),
        choices=list(range(1, 8)),
        help=("which spectral views from cache_combined to pay for when the gate "
              "fires. 1..7 map to cutoffs 0.90,0.75,0.60,0.45,0.30,0.20,0.12. "
              "Example: --spectral-views 4 5 6 tests cutoffs .45/.30/.20."),
    )
    ap.add_argument("--spatial-dir", default="cache_spatial")
    ap.add_argument("--gate-cache", default="cache_gate",
                    help="small cache for model-free FFT gate scores")
    ap.add_argument("--tuning", default="auto",
                    help="'auto' or a tuning JSON path (single-model runs only)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tune-severities", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--eval-severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--hf-radii", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6],
                    help="FFT radii whose outer-band energy fraction is tried as the gate")
    ap.add_argument("--budgets", nargs="+", type=float, default=None,
                    help=("maximum average classifier forwards used when tuning the gate; "
                          "default is every integer from 8 through the expensive-branch cost"))
    ap.add_argument("--n-thresholds", type=int, default=101,
                    help="number of quantile thresholds per FFT radius")
    ap.add_argument("--fft-batch-size", type=int, default=128)
    ap.add_argument("--image-workers", type=int, default=8)
    ap.add_argument("--device", default=None,
                    help="device for FFT gate feature computation; default cuda/mps/cpu")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    spectral_views = sorted(set(args.spectral_views))
    extra_spectral_forwards = len(spectral_views)
    branch_forwards = SPATIAL_FORWARDS + extra_spectral_forwards
    if args.budgets is None:
        args.budgets = list(range(SPATIAL_FORWARDS, branch_forwards + 1))
    if any(b < SPATIAL_FORWARDS or b > branch_forwards for b in args.budgets):
        raise SystemExit(
            f"--budgets must lie in [{SPATIAL_FORWARDS}, {branch_forwards}] "
            f"for spectral views {spectral_views}"
        )
    if args.tuning != "auto" and len(args.models) != 1:
        raise SystemExit("An explicit --tuning path is only supported with one --models entry")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print("FFT gate device:", device)
    print("radii:", args.hf_radii, "| budgets:", args.budgets)
    print(
        "spectral views:", spectral_views,
        "cutoffs:", [SPECTRAL_CUTOFFS[v] for v in spectral_views],
        "| expensive branch forwards:", branch_forwards,
    )
    if spectral_views != list(range(1, 8)):
        print(
            "NOTE: reduced spectral subsets reuse the fusion coefficients tuned "
            "for the full combined profile. Treat this as a screening experiment; "
            "retune the winning reduced profile before using it as a final result."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_model_summaries = {}

    for model_name in args.models:
        tuning_path = (
            latest_tuning(model_name, "cache_combined", args.results_dir)
            if args.tuning == "auto" else Path(args.tuning)
        )
        tuning = load_tuning(tuning_path)
        print(f"\n=== {model_name} | tuning: {tuning_path.name} ===")

        # budget -> all held-out-family evaluation rows
        eval_rows_by_budget = {float(b): [] for b in args.budgets}
        fold_records = {}

        for held, tune_corrs, test_corrs in leave_one_family_out():
            cfg = fold_config(tuning, held)
            print(
                f"\n  hold out {held}: cfg a={cfg.alpha} b={cfg.beta} g={cfg.gamma} "
                f"tau={cfg.tau} std={cfg.standardize} m={cfg.m} agree={cfg.agreement}"
            )

            tune_conditions = []
            test_conditions = []

            # Build tune-side per-image predictions/features.
            for corr in tune_corrs:
                for sev in args.tune_severities:
                    cache.CACHE_ROOT = Path(args.combined_dir)
                    if not cache.has_condition(model_name, corr, sev):
                        continue
                    d = cache.load_condition(model_name, corr, sev)
                    tune_mask, _ = tune_test_masks(d["paths"], args.frac_tune, args.split_seed)
                    cp, sp, labels, paths = condition_predictions(
                        model_name, corr, sev, cfg, tune_mask,
                        combined_dir=args.combined_dir, spatial_dir=args.spatial_dir,
                        spectral_views=spectral_views,
                    )
                    gate_all = load_or_compute_gate_scores(
                        d["paths"], corr, sev, args.hf_radii,
                        gate_cache=args.gate_cache, device=device,
                        batch_size=args.fft_batch_size, workers=args.image_workers,
                    )
                    tune_conditions.append({
                        "corruption": corr, "severity": sev,
                        "combined_pred": cp, "spatial_pred": sp,
                        "labels": labels, "paths": paths,
                        "gate": gate_all[tune_mask],
                    })

            if not tune_conditions:
                raise SystemExit(f"No tuning conditions available for fold {held}")

            # One threshold curve per candidate FFT radius.
            curves = {}
            for ridx, radius in enumerate(args.hf_radii):
                curves[float(radius)] = threshold_curve(
                    tune_conditions, ridx, extra_spectral_forwards, n_thresholds=args.n_thresholds
                )
            policies = choose_policy(curves, args.budgets)

            for budget in args.budgets:
                p = policies[float(budget)]
                print(
                    f"    budget {budget:>4.1f}: r={p['radius']:.2f} "
                    f"thr={p['threshold']:.6f} tune={100*p['tune_acc']:.2f}% "
                    f"gate={100*p['gate_rate']:.1f}% fwds={p['avg_forwards']:.2f}"
                )

            # Build held-out test-side predictions/features once.
            for corr in test_corrs:
                for sev in args.eval_severities:
                    cache.CACHE_ROOT = Path(args.combined_dir)
                    if not cache.has_condition(model_name, corr, sev):
                        continue
                    d = cache.load_condition(model_name, corr, sev)
                    _, test_mask = tune_test_masks(d["paths"], args.frac_tune, args.split_seed)
                    cp, sp, labels, paths = condition_predictions(
                        model_name, corr, sev, cfg, test_mask,
                        combined_dir=args.combined_dir, spatial_dir=args.spatial_dir,
                        spectral_views=spectral_views,
                    )
                    gate_all = load_or_compute_gate_scores(
                        d["paths"], corr, sev, args.hf_radii,
                        gate_cache=args.gate_cache, device=device,
                        batch_size=args.fft_batch_size, workers=args.image_workers,
                    )
                    test_conditions.append({
                        "corruption": corr, "severity": sev,
                        "combined_pred": cp, "spatial_pred": sp,
                        "labels": labels, "paths": paths,
                        "gate": gate_all[test_mask],
                    })

            fold_records[held] = {"policies": {str(b): policies[float(b)] for b in args.budgets}}

            for budget in args.budgets:
                p = policies[float(budget)]
                ridx = args.hf_radii.index(p["radius"])
                rows = evaluate_policy(test_conditions, ridx, p["threshold"], extra_spectral_forwards)
                for r in rows:
                    r.update(model=model_name, held_out_family=held, budget=float(budget),
                             radius=p["radius"], threshold=p["threshold"])
                eval_rows_by_budget[float(budget)].extend(rows)

        # Aggregate across the four held-out families.
        summaries = {}
        print(f"\n  {'budget':>6} {'spatial':>8} {'gated':>8} {'combined':>9} "
              f"{'gain-vs-sp':>10} {'gate%':>7} {'fwds':>7}")
        print("  " + "-" * 62)
        for budget in args.budgets:
            rows = eval_rows_by_budget[float(budget)]
            gate_rate = weighted_gate_rate(rows)
            summary = {
                "budget": float(budget),
                "mCA_spatial": mean_field(rows, "spatial_acc"),
                "mCA_gated": mean_field(rows, "gated_acc"),
                "mCA_combined": mean_field(rows, "combined_acc"),
                "gate_rate": gate_rate,
                "avg_forwards": float(SPATIAL_FORWARDS + extra_spectral_forwards * gate_rate),
            }
            summary["gain_vs_spatial_pts"] = 100 * (
                summary["mCA_gated"] - summary["mCA_spatial"]
            )
            summary["gap_to_full_combined_pts"] = 100 * (
                summary["mCA_combined"] - summary["mCA_gated"]
            )
            summaries[str(float(budget))] = summary
            print(
                f"  {budget:6.1f} {100*summary['mCA_spatial']:8.2f} "
                f"{100*summary['mCA_gated']:8.2f} {100*summary['mCA_combined']:9.2f} "
                f"{summary['gain_vs_spatial_pts']:+10.2f} "
                f"{100*gate_rate:7.1f} {summary['avg_forwards']:7.2f}"
            )

        # Per-condition CSV for plotting / bootstrap follow-up.
        csv_path = out_dir / f"gated_combined_{model_name}_{stamp}.csv"
        fieldnames = [
            "model", "held_out_family", "budget", "radius", "threshold",
            "corruption", "severity", "n", "spatial_acc", "gated_acc",
            "combined_acc", "gate_rate", "avg_forwards",
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for budget in args.budgets:
                for row in eval_rows_by_budget[float(budget)]:
                    w.writerow(row)

        result = {
            "model": model_name,
            "timestamp": stamp,
            "tuning_file": str(tuning_path),
            "combined_dir": args.combined_dir,
            "spectral_views": spectral_views,
            "spectral_cutoffs": [SPECTRAL_CUTOFFS[v] for v in spectral_views],
            "expensive_branch_forwards": branch_forwards,
            "spatial_dir": args.spatial_dir,
            "gate_cache": args.gate_cache,
            "hf_radii": args.hf_radii,
            "budgets": args.budgets,
            "tune_severities": args.tune_severities,
            "eval_severities": args.eval_severities,
            "split": {"frac_tune": args.frac_tune, "seed": args.split_seed},
            "folds": fold_records,
            "summaries": summaries,
        }
        json_path = out_dir / f"gated_combined_{model_name}_{stamp}.json"
        json_path.write_text(json.dumps(result, indent=2))
        print(f"  saved {csv_path}")
        print(f"  saved {json_path}")
        all_model_summaries[model_name] = summaries

    # A compact cross-model JSON is convenient for report plotting.
    combined_out = out_dir / f"gated_combined_summary_{stamp}.json"
    combined_out.write_text(json.dumps(all_model_summaries, indent=2))
    print("\nsaved", combined_out)


if __name__ == "__main__":
    main()

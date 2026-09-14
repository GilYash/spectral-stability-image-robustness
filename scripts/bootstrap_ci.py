"""Paired bootstrap confidence intervals for the headline deltas: does fusion
really beat the original classifier, and how does it really compare to the
spatial-TTA baseline? evaluate.py reports point estimates only; this adds
error bars to the same statistic (mean accuracy over corruptions x severities,
each condition itself a mean over its held-out images).

Resampling is paired and within-condition: for each of the 75 (corruption,
severity) conditions, images are bootstrap-resampled (with replacement, same
N), and the SAME resampled indices are used for every predictor being
compared, so the bootstrap targets the paired difference directly rather than
two independent CIs that would overstate the uncertainty on their gap.

    python scripts/bootstrap_ci.py                      # both models, cache_fixed
    python scripts/bootstrap_ci.py --models resnet18 --n-boot 5000
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import cache
from src.metrics import FusionConfig, precompute, fused_probs, top1
from src.splits import CORRUPTION_FAMILY, tune_test_masks


def latest_tuning(model_name, tag):
    """Path of the newest results/tuning_<model>[_<tag>]_*.json, or exit."""
    pattern = f"results/tuning_{model_name}_{tag}_*.json" if tag else f"results/tuning_{model_name}_2*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no {pattern} found -- run tune_fusion.py first")
    return files[-1]


def config_picker(model_name, tag):
    """Build the corruption -> FusionConfig mapping from the newest tuning
    json (same held-out-family-or-global logic as evaluate.py's).

    Returns:
        (pick, path) where `pick(corruption)` -> FusionConfig and `path` is
        the tuning json it came from.
    """
    path = latest_tuning(model_name, tag)
    tune = json.loads(Path(path).read_text())
    held = {fam: FusionConfig.from_dict(tune["held_out"][fam]["config"])
            for fam in tune["held_out"]}
    glob_cfg = FusionConfig.from_dict(tune["global"]["config"])

    def pick(corruption):
        fam = CORRUPTION_FAMILY.get(corruption)
        return held.get(fam, glob_cfg)

    return pick, path


def per_image_correctness(model_name, cache_dir, tag, ladder_len, severities,
                          frac_tune, split_seed):
    """orig/fused (and, if cache_spatial exists, tta) per-image correctness,
    one (bool array, n_images) per condition, aligned across predictors.

    Returns:
        list of dicts, one per condition: {'orig': bool[N], 'fused': bool[N],
        'tta': bool[N] or None, 'n': N}.
    """
    cache.CACHE_ROOT = Path(cache_dir)
    pick, tuning_path = config_picker(model_name, tag)

    spatial_root = Path("cache_spatial")
    has_spatial = spatial_root.exists()

    conditions = []
    for corruption in CORRUPTION_FAMILY:
        for sev in severities:
            if not cache.has_condition(model_name, corruption, sev):
                continue
            d = cache.load_condition(model_name, corruption, sev)
            _, mask = tune_test_masks(d["paths"], frac_tune, split_seed)
            cfg = pick(corruption)
            probs = d["probs"][mask].astype(np.float32)
            labels = d["labels"][mask]
            pre = precompute(probs, m=cfg.m, agreement=cfg.agreement, ladder_len=ladder_len)
            orig_correct = top1(probs[:, 0]) == labels
            fused_correct = top1(fused_probs(pre, cfg)) == labels

            tta_correct = None
            if has_spatial:
                cache.CACHE_ROOT = spatial_root
                if cache.has_condition(model_name, corruption, sev):
                    ds = cache.load_spatial_condition(model_name, corruption, sev)
                    paths = list(d["paths"])
                    pos = {p: i for i, p in enumerate(ds["paths"])}
                    idx = [pos[p] for p in paths]
                    sp = ds["probs"][idx][mask].astype(np.float32)
                    tta_correct = top1(sp.mean(axis=1)) == labels
                cache.CACHE_ROOT = Path(cache_dir)

            conditions.append({"orig": orig_correct, "fused": fused_correct,
                               "tta": tta_correct, "n": int(mask.sum())})
    return conditions, tuning_path


def bootstrap_delta(conditions, key_a, key_b, n_boot, rng):
    """Bootstrap replicates of mean_cond(mean_img(key_a)) - mean_cond(mean_img(key_b)).

    Only uses conditions where both keys are available (skips e.g. 'tta' if
    cache_spatial is missing a condition).
    """
    usable = [c for c in conditions if c[key_a] is not None and c[key_b] is not None]
    if not usable:
        return None
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        cond_means_a = np.empty(len(usable))
        cond_means_b = np.empty(len(usable))
        for i, c in enumerate(usable):
            idx = rng.integers(0, c["n"], size=c["n"])
            cond_means_a[i] = c[key_a][idx].mean()
            cond_means_b[i] = c[key_b][idx].mean()
        deltas[b] = cond_means_a.mean() - cond_means_b.mean()
    return deltas


def main():
    """Compute and print the bootstrap CIs for one cache, both models."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"],
                    choices=["resnet18", "deit"])
    ap.add_argument("--cache-dir", default="cache_fixed")
    ap.add_argument("--tag", default="cache_fixed")
    ap.add_argument("--ladder-len", type=int, default=None)
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for model_name in args.models:
        print(f"\n=== {model_name}  ({args.cache_dir}) ===")
        conditions, tuning_path = per_image_correctness(
            model_name, args.cache_dir, args.tag, args.ladder_len,
            args.severities, args.frac_tune, args.split_seed,
        )
        print(f"  {len(conditions)} conditions, config: {tuning_path}")

        model_report = {"tuning_config": tuning_path, "n_boot": args.n_boot}
        for key_b, label in [("orig", "fused - orig"), ("tta", "fused - spatial TTA")]:
            deltas = bootstrap_delta(conditions, "fused", key_b, args.n_boot, rng)
            if deltas is None:
                print(f"  {label:22s}: n/a (no spatial cache)")
                continue
            point = np.mean([c["fused"].mean() for c in conditions if c[key_b] is not None]) - \
                    np.mean([c[key_b].mean() for c in conditions if c[key_b] is not None])
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            p_le_zero = float(np.mean(deltas <= 0))
            print(f"  {label:22s}: {point*100:+.2f} pts   95% CI [{lo*100:+.2f}, {hi*100:+.2f}]   "
                  f"P(delta <= 0) = {p_le_zero:.3f}")
            model_report[label] = {"point": float(point), "ci95": [float(lo), float(hi)],
                                   "p_le_zero": p_le_zero}
        report[model_name] = model_report

    path = out_dir / f"bootstrap_ci_{args.tag}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()

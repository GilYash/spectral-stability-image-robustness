"""Tune the fusion coefficients (alpha, beta, gamma, tau, standardize) by grid
search over the cached profiles.

Two things keep the numbers honest:
  * leave-one-corruption-family-out -- each family gets a config that was tuned
    without ever seeing that family, so evaluate.py can report a real
    generalisation number for it;
  * a disjoint image half -- tuning uses one half of the images, evaluation the
    other (see src/splits.py).

Writes one results/tuning_<model>_<timestamp>.json per model, holding the best
config for each held-out family plus a "global" config tuned on all families.

    python scripts/tune_fusion.py                       # both models
    python scripts/tune_fusion.py --models resnet18 --tune-severities 3
"""
import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import cache
from src.metrics import FusionConfig, precompute, score
from src.splits import FAMILIES, CORRUPTION_FAMILY, leave_one_family_out, tune_test_masks

# --- the search grid (dropping the all-zero-weight config) -------------------
ALPHAS = (0.0, 1.0)
BETAS = (0.0, 0.25, 0.5, 1.0)
GAMMAS = (0.0, 0.25, 0.5, 1.0)
TAUS = (1.0, 0.5, 0.2, 0.1, 0.05)
STANDARDIZE = (True, False)


def grid(m):
    """Yield every FusionConfig in the search grid (fixed `m`), ~310 of them."""
    for a, b, g, tau, std in itertools.product(ALPHAS, BETAS, GAMMAS, TAUS, STANDARDIZE):
        if a == 0.0 and b == 0.0 and g == 0.0:
            continue
        yield FusionConfig(alpha=a, beta=b, gamma=g, tau=tau, standardize=std, m=m)


def best_config(configs, conditions):
    """Pick the config with the highest mean fused accuracy over `conditions`.

    Args:
        configs: iterable of FusionConfig.
        conditions: list of (precomputed, labels) pairs from `precompute()`.

    Returns:
        (best_config, mean_fused_acc), or (None, nan) if `conditions` is empty.
    """
    if not conditions:
        return None, float("nan")
    best, best_acc = None, -1.0
    for cfg in configs:
        acc = np.mean([score(pre, lab, cfg, full=False) for pre, lab in conditions])
        if acc > best_acc:
            best, best_acc = cfg, float(acc)
    return best, best_acc


def baseline_means(conditions):
    """Mean original-view and equal-weight accuracy over `conditions` (for context)."""
    orig = np.mean([np.mean(pre["orig"].argmax(-1) == lab) for pre, lab in conditions])
    equal = np.mean([np.mean(pre["equal_pred"] == lab) for pre, lab in conditions])
    return float(orig), float(equal)


def main():
    """Precompute the tune-half signals per model, then run the grid per fold."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"],
                    choices=["resnet18", "deit"])
    ap.add_argument("--tune-severities", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache-dir", default="cache",
                    help="read profiles from this cache (must match cache_profiles.py --cache-dir)")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output filename, e.g. the ladder name; "
                         "defaults to the cache dir name when it isn't 'cache'")
    args = ap.parse_args()

    cache.CACHE_ROOT = Path(args.cache_dir)
    tag = args.tag or (Path(args.cache_dir).name if args.cache_dir != "cache" else None)

    configs = list(grid(args.m))
    print(f"grid: {len(configs)} configs | tune severities {args.tune_severities} "
          f"| cache: {args.cache_dir}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        print(f"\n=== {model_name} ===")

        # precompute signals for every (corruption, tune-severity) on the tune half
        pre_by_cond = {}
        t0 = time.time()
        for corruption in CORRUPTION_FAMILY:
            for sev in args.tune_severities:
                if not cache.has_condition(model_name, corruption, sev):
                    print(f"  missing cache: {corruption}/{sev}, skipping")
                    continue
                d = cache.load_condition(model_name, corruption, sev)
                tune_mask, _ = tune_test_masks(d["paths"], args.frac_tune, args.split_seed)
                pre = precompute(d["probs"][tune_mask], m=args.m)
                pre_by_cond[(corruption, sev)] = (pre, d["labels"][tune_mask])
        print(f"  precomputed {len(pre_by_cond)} conditions in {time.time() - t0:.0f}s")

        def conds_for(corruptions):
            """The (precomputed, labels) pairs for these corruptions x tune severities."""
            return [pre_by_cond[(c, s)] for c in corruptions for s in args.tune_severities
                    if (c, s) in pre_by_cond]

        record = {
            "model": model_name,
            "cache_dir": args.cache_dir,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "grid_size": len(configs),
            "tune_severities": args.tune_severities,
            "split": {"frac_tune": args.frac_tune, "seed": args.split_seed},
            "m": args.m,
            "held_out": {},
            "global": {},
        }

        def tune_on(name, tune_corr):
            """Grid-search on `tune_corr`, print the winner, return its record dict."""
            conds = conds_for(tune_corr)
            if not conds:
                print(f"  {name:14s}: no cached tune conditions, skipping")
                return None
            t0 = time.time()
            cfg, acc = best_config(configs, conds)
            orig, equal = baseline_means(conds)
            print(f"  {name:14s}: fused {acc*100:.1f}%  "
                  f"(orig {orig*100:.1f}, equal {equal*100:.1f})  "
                  f"a={cfg.alpha} b={cfg.beta} g={cfg.gamma} tau={cfg.tau} "
                  f"std={cfg.standardize}  [{time.time()-t0:.0f}s]")
            return {"config": cfg.as_dict(), "mean_fused_acc": acc,
                    "mean_orig_acc": orig, "mean_equal_acc": equal}

        for held, tune_corr, _ in leave_one_family_out():
            entry = tune_on(f"hold out {held}", tune_corr)
            if entry:
                record["held_out"][held] = entry
        record["global"] = tune_on("global", list(CORRUPTION_FAMILY)) or {}

        name = f"tuning_{model_name}" + (f"_{tag}" if tag else "") + f"_{record['timestamp']}.json"
        out = out_dir / name
        out.write_text(json.dumps(record, indent=2))
        print("  saved", out)


if __name__ == "__main__":
    main()

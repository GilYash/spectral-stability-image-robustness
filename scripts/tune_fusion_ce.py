"""Fit non-negative alpha/beta/gamma by cross-entropy on the tune half.

This is the continuous alternative to scripts/tune_fusion.py's coefficient
Grid.  It preserves the same experimental discipline:

- same deterministic tune/test image split
- leave-one-corruption-family-out configurations
- m in {1,2,3} searched outside the continuous optimizer
- agreement in {all, origin} searched outside the continuous optimizer
- signals are standardized per image
- tau is fixed to 1; scale is absorbed into alpha/beta/gamma
- alpha,beta,gamma are constrained to be >= 0

The output uses the same held_out/global config structure as tune_fusion.py, so
scripts/evaluate.py can consume it with --tuning <path>.

Examples:
    python scripts/tune_fusion_ce.py --cache-dir cache_fixed --grid-tuning auto
    python scripts/tune_fusion_ce.py --cache-dir cache_geometric --grid-tuning auto
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import analysis, cache
from src.ce_fusion import DEFAULT_EPS, fit_nonnegative, loss_and_grad
from src.metrics import FusionConfig
from src.splits import CORRUPTION_FAMILY, leave_one_family_out, tune_test_masks

M_VALUES = (1, 2, 3)
AGREEMENT_MODES = ("all", "origin")


def latest_grid_tuning(model_name, tag, results_dir="results"):
    """Newest *grid* tuning file for this model/cache tag."""
    pattern = str(Path(results_dir) / f"tuning_{model_name}_{tag}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[-1]


def load_grid_record(spec, model_name, tag, results_dir):
    """Load an optional grid result used only as an optimizer start."""
    if not spec:
        return None, None
    path = latest_grid_tuning(model_name, tag, results_dir) if spec == "auto" else spec
    if path is None:
        print("  grid start: none found; using generic/random starts")
        return None, None
    rec = json.loads(Path(path).read_text())
    print("  grid start:", path)
    return rec, path


def grid_theta_start(grid_record, held_family=None):
    if not grid_record:
        return None
    try:
        entry = (grid_record["global"] if held_family is None
                 else grid_record["held_out"][held_family])
        c = entry["config"]
        tau = float(c.get("tau", 1.0))
        return np.array([c["alpha"], c["beta"], c["gamma"]], dtype=float) / tau
    except (KeyError, TypeError):
        return None


def precompute_signal_variants(probs, labels, m_values, agreement_modes,
                               ladder_len=None, chunk=256):
    """Compute all (m, agreement) signal matrices with one JSD pass.

    Returns:
        variants: {(m, agreement): X}, X shape (N, V, 3), standardized.
        p_true: (N, V), true-class probability for every view.
    """
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    n, v, _ = probs.shape

    C = analysis.confidence(probs).astype(np.float32)
    A = {a: np.empty((n, v), dtype=np.float32) for a in agreement_modes}
    S = {m: np.empty((n, v), dtype=np.float32) for m in m_values}

    for i in range(0, n, chunk):
        p = probs[i:i + chunk]
        d = analysis.pairwise_jsd(p)
        if "all" in A:
            A["all"][i:i + len(p)] = analysis.agreement_from_d(d)
        if "origin" in A:
            # Exactly -JSD(p_k, p_0), reusing the pairwise matrix.
            A["origin"][i:i + len(p)] = -d[:, :, 0]
        for m in m_values:
            S[m][i:i + len(p)] = analysis.stability_from_d(
                d, m=m, ladder_len=ladder_len
            )

    C = analysis.zscore(C).astype(np.float32)
    A = {k: analysis.zscore(x).astype(np.float32) for k, x in A.items()}
    S = {k: analysis.zscore(x).astype(np.float32) for k, x in S.items()}

    variants = {}
    for m in m_values:
        for agreement in agreement_modes:
            variants[(m, agreement)] = np.stack(
                [C, A[agreement], S[m]], axis=-1
            ).astype(np.float32)

    p_true = np.take_along_axis(
        probs, labels[:, None, None], axis=2
    ).squeeze(2).astype(np.float32)
    return variants, p_true


def concatenate_conditions(by_key, corruptions, severities, key):
    xs, ps = [], []
    for c in corruptions:
        for s in severities:
            rec = by_key[key].get((c, s))
            if rec is not None:
                xs.append(rec["X"])
                ps.append(rec["p_true"])
    if not xs:
        return None, None
    return np.concatenate(xs, axis=0), np.concatenate(ps, axis=0)


def baseline_means(base_meta, corruptions, severities):
    recs = [base_meta[(c, s)] for c in corruptions for s in severities
            if (c, s) in base_meta]
    if not recs:
        return float("nan"), float("nan")
    return (float(np.mean([r["orig_acc"] for r in recs])),
            float(np.mean([r["equal_acc"] for r in recs])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"],
                    choices=["resnet18", "deit"])
    ap.add_argument("--tune-severities", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--m-values", nargs="+", type=int, default=list(M_VALUES))
    ap.add_argument("--agreement-modes", nargs="+", choices=["all", "origin"],
                    default=list(AGREEMENT_MODES))
    ap.add_argument("--ladder-len", type=int, default=None,
                    help="ordered-ladder length for combined profiles; None for spectral-only")
    ap.add_argument("--cache-dir", default="cache_fixed")
    ap.add_argument("--tag", default=None,
                    help="output/cache tag; defaults to cache directory name")
    ap.add_argument("--out", default="results")
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS,
                    help="floor for cached float16 true-class probabilities")
    ap.add_argument("--random-restarts", type=int, default=4)
    ap.add_argument("--optimizer-seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--signal-chunk", type=int, default=256,
                    help="images per pairwise-JSD chunk")
    ap.add_argument("--grid-tuning", default="auto",
                    help="grid JSON to use as one optimizer start, 'auto', or empty string")
    args = ap.parse_args()

    cache.CACHE_ROOT = Path(args.cache_dir)
    tag = args.tag or Path(args.cache_dir).name
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [(m, a) for m in args.m_values for a in args.agreement_modes]

    print(
        f"CE fit: theta=(alpha,beta,gamma)>=0, tau=1, std=True, l2={args.l2:g} | "
        f"{len(keys)} (m,agreement) combos | cache: {args.cache_dir}"
    )

    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        grid_record, grid_path = load_grid_record(
            args.grid_tuning, model_name, tag, args.out
        )

        # Small in-memory representation: X(N,V,3) + p_k(y), not all 1000 probs.
        by_key = {key: {} for key in keys}
        base_meta = {}
        t0 = time.time()
        n_conditions = 0

        for corruption in CORRUPTION_FAMILY:
            for sev in args.tune_severities:
                if not cache.has_condition(model_name, corruption, sev):
                    print(f"  missing cache: {corruption}/{sev}, skipping")
                    continue

                d = cache.load_condition(model_name, corruption, sev)
                tune_mask, _ = tune_test_masks(
                    d["paths"], args.frac_tune, args.split_seed
                )
                probs = d["probs"][tune_mask]
                labels = d["labels"][tune_mask]

                variants, p_true = precompute_signal_variants(
                    probs, labels,
                    m_values=args.m_values,
                    agreement_modes=args.agreement_modes,
                    ladder_len=args.ladder_len,
                    chunk=args.signal_chunk,
                )
                for key, X in variants.items():
                    by_key[key][(corruption, sev)] = {
                        "X": X,
                        "p_true": p_true,
                    }

                p32 = probs.astype(np.float32)
                base_meta[(corruption, sev)] = {
                    "orig_acc": float(np.mean(p32[:, 0].argmax(-1) == labels)),
                    "equal_acc": float(np.mean(p32.mean(axis=1).argmax(-1) == labels)),
                }
                n_conditions += 1

        print(
            f"  precomputed {n_conditions} tune conditions; all signal variants in "
            f"{time.time() - t0:.0f}s"
        )

        record = {
            "model": model_name,
            "method": "cross_entropy_nonnegative",
            "cache_dir": args.cache_dir,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "tune_severities": args.tune_severities,
            "split": {"frac_tune": args.frac_tune, "seed": args.split_seed},
            "m_values": args.m_values,
            "ladder_len": args.ladder_len,
            "agreement_modes": args.agreement_modes,
            "standardize": True,
            "tau": 1.0,
            "nonnegative": True,
            "l2": args.l2,
            "eps": args.eps,
            "random_restarts": args.random_restarts,
            "grid_start": grid_path,
            "held_out": {},
            "global": {},
        }

        def tune_on(name, tune_corr, held_family=None):
            start = grid_theta_start(grid_record, held_family)
            best = None
            t1 = time.time()

            for key in keys:
                X, p_true = concatenate_conditions(
                    by_key, tune_corr, args.tune_severities, key
                )
                if X is None:
                    continue

                starts = [start] if start is not None else None
                theta, objective, res = fit_nonnegative(
                    X, p_true,
                    l2=args.l2,
                    eps=args.eps,
                    starts=starts,
                    random_restarts=args.random_restarts,
                    seed=args.optimizer_seed,
                    maxiter=args.maxiter,
                )
                nll, _ = loss_and_grad(
                    theta, X, p_true, l2=0.0, eps=args.eps
                )
                candidate = {
                    "key": key,
                    "theta": theta,
                    "objective": float(objective),
                    "nll": float(nll),
                    "success": bool(res.success),
                    "message": str(res.message),
                    "nit": int(res.nit),
                    "n_samples": int(len(X)),
                }
                if best is None or candidate["objective"] < best["objective"]:
                    best = candidate

            if best is None:
                print(f"  {name:14s}: no tune data")
                return None

            m, agreement = best["key"]
            a, b, g = best["theta"]
            cfg = FusionConfig(
                alpha=float(a), beta=float(b), gamma=float(g),
                tau=1.0, standardize=True, m=int(m), agreement=agreement,
            )
            orig, equal = baseline_means(
                base_meta, tune_corr, args.tune_severities
            )
            print(
                f"  {name:14s}: NLL {best['nll']:.5f}  obj {best['objective']:.5f}  "
                f"orig {orig*100:.1f}% equal {equal*100:.1f}%  "
                f"a={cfg.alpha:.4f} b={cfg.beta:.4f} g={cfg.gamma:.4f} "
                f"m={cfg.m} agree={cfg.agreement}  [{time.time()-t1:.1f}s]"
            )
            return {
                "config": cfg.as_dict(),
                # Kept for compatibility with the old tuning JSON structure.
                # CE chooses by NLL, so tune accuracy is intentionally not used.
                "mean_fused_acc": None,
                "mean_orig_acc": orig,
                "mean_equal_acc": equal,
                "mean_nll": best["nll"],
                "objective": best["objective"],
                "optimizer": {
                    "success": best["success"],
                    "message": best["message"],
                    "nit": best["nit"],
                    "n_samples": best["n_samples"],
                },
            }

        for held, tune_corr, _ in leave_one_family_out():
            entry = tune_on(f"hold out {held}", tune_corr, held_family=held)
            if entry:
                record["held_out"][held] = entry

        record["global"] = tune_on(
            "global", list(CORRUPTION_FAMILY), held_family=None
        ) or {}

        out = out_dir / f"tuning_ce_{model_name}_{tag}_{record['timestamp']}.json"
        out.write_text(json.dumps(record, indent=2))
        print("  saved", out)


if __name__ == "__main__":
    main()

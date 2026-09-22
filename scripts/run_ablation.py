"""Orchestrate the 2x2 spectral ablation.

Cells:
    fixed      x grid
    fixed      x cross-entropy
    geometric  x grid
    geometric  x cross-entropy

The script reuses tuning files when present, runs fresh evaluation with explicit
config paths, and writes one compact CSV/JSON summary.  It does NOT silently
start the expensive geometric GPU cache unless --build-geometric-cache is
passed.

Typical flow:
    # 1) after installing the patch, build the new cache once
    python scripts/run_ablation.py --build-geometric-cache

    # 2) later reruns reuse caches/tuning where possible
    python scripts/run_ablation.py
"""

import argparse
import csv
import glob
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def latest(pattern):
    files = sorted(glob.glob(str(pattern)))
    return files[-1] if files else None


def run(cmd):
    print("\n$", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)


def cache_has_npz(path, model):
    return any((Path(path) / model).rglob("*.npz")) if (Path(path) / model).exists() else False


def ensure_grid_tuning(py, model, cache_dir, results_dir, tune_severities, force=False):
    tag = Path(cache_dir).name
    pattern = Path(results_dir) / f"tuning_{model}_{tag}_*.json"
    found = latest(pattern)
    if found and not force:
        print(f"reuse grid tuning: {found}")
        return found

    cmd = [
        py, "scripts/tune_fusion.py",
        "--models", model,
        "--cache-dir", cache_dir,
        "--tag", tag,
        "--out", results_dir,
        "--tune-severities", *map(str, tune_severities),
    ]
    run(cmd)
    found = latest(pattern)
    if not found:
        raise RuntimeError(f"grid tuner finished but no file matched {pattern}")
    return found


def ensure_ce_tuning(py, model, cache_dir, results_dir, tune_severities,
                     grid_path, l2, force=False):
    tag = Path(cache_dir).name
    pattern = Path(results_dir) / f"tuning_ce_{model}_{tag}_*.json"
    found = latest(pattern)
    if found and not force:
        print(f"reuse CE tuning: {found}")
        return found

    cmd = [
        py, "scripts/tune_fusion_ce.py",
        "--models", model,
        "--cache-dir", cache_dir,
        "--tag", tag,
        "--out", results_dir,
        "--tune-severities", *map(str, tune_severities),
        "--l2", str(l2),
        "--grid-tuning", grid_path,
    ]
    run(cmd)
    found = latest(pattern)
    if not found:
        raise RuntimeError(f"CE tuner finished but no file matched {pattern}")
    return found


def run_evaluation(py, model, cache_dir, tuning_path, result_tag, results_dir):
    """Run evaluation fresh so all cells use the same metric implementation."""
    before = set(glob.glob(str(Path(results_dir) / f"eval_{model}_{result_tag}_*.json")))
    run([
        py, "scripts/evaluate.py",
        "--models", model,
        "--cache-dir", cache_dir,
        "--tuning", tuning_path,
        "--tag", result_tag,
        "--out", results_dir,
    ])
    after = set(glob.glob(str(Path(results_dir) / f"eval_{model}_{result_tag}_*.json")))
    new = sorted(after - before)
    path = new[-1] if new else latest(Path(results_dir) / f"eval_{model}_{result_tag}_*.json")
    if not path:
        raise RuntimeError(f"evaluate.py produced no result for {model}/{result_tag}")
    return path


def row_from_eval(model, ladder, fitter, cache_dir, tuning_path, eval_path):
    e = json.loads(Path(eval_path).read_text())
    t = json.loads(Path(tuning_path).read_text())
    cfg = t.get("global", {}).get("config", {})
    return {
        "model": model,
        "ladder": ladder,
        "fitter": fitter,
        "cache_dir": cache_dir,
        "tuning": tuning_path,
        "evaluation": eval_path,
        "alpha_global": cfg.get("alpha"),
        "beta_global": cfg.get("beta"),
        "gamma_global": cfg.get("gamma"),
        "tau_global": cfg.get("tau", 1.0),
        "alpha_over_tau_global": (cfg.get("alpha") / cfg.get("tau", 1.0)
                                  if cfg.get("alpha") is not None else None),
        "beta_over_tau_global": (cfg.get("beta") / cfg.get("tau", 1.0)
                                 if cfg.get("beta") is not None else None),
        "gamma_over_tau_global": (cfg.get("gamma") / cfg.get("tau", 1.0)
                                  if cfg.get("gamma") is not None else None),
        "m_global": cfg.get("m"),
        "agreement_global": cfg.get("agreement"),
        "held_out_configs": {k: v.get("config", {}) for k, v in t.get("held_out", {}).items()},
        "config_note": "corrupted evaluation uses family-specific held-out configs; global is used for clean",
        "mCA_orig": e.get("mCA_orig"),
        "mCA_fused": e.get("mCA_fused"),
        "gain_pts": (100.0 * (e["mCA_fused"] - e["mCA_orig"])
                     if e.get("mCA_fused") is not None and e.get("mCA_orig") is not None else None),
        "mean_orig_nll": e.get("mean_orig_nll"),
        "mean_fused_nll": e.get("mean_fused_nll"),
        "mean_orig_ece": e.get("mean_orig_ece"),
        "mean_fused_ece": e.get("mean_fused_ece"),
        "negative_transfer": e.get("mean_negative_transfer"),
        "clean_orig_acc": e.get("clean_orig_acc"),
        "clean_fused_acc": e.get("clean_fused_acc"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"],
                    choices=["resnet18", "deit"])
    ap.add_argument("--fixed-cache", default="cache_fixed")
    ap.add_argument("--geometric-cache", default="cache_geometric")
    ap.add_argument("--results", default="results")
    ap.add_argument("--tune-severities", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--build-geometric-cache", action="store_true",
                    help="if geometric cache is missing, run the expensive GPU cache step")
    ap.add_argument("--force-tuning", action="store_true",
                    help="rerun grid and CE tuning even if matching JSONs exist")
    args = ap.parse_args()

    py = sys.executable
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    missing_geom = [m for m in args.models if not cache_has_npz(args.geometric_cache, m)]
    if missing_geom:
        if not args.build_geometric_cache:
            raise SystemExit(
                "geometric cache is missing for: " + ", ".join(missing_geom) + "\n"
                "Build it explicitly with:\n"
                f"  {py} scripts/cache_profiles.py --ladder geometric "
                f"--cache-dir {args.geometric_cache} --models " + " ".join(missing_geom) + "\n"
                "or rerun this script with --build-geometric-cache."
            )
        run([
            py, "scripts/cache_profiles.py",
            "--ladder", "geometric",
            "--cache-dir", args.geometric_cache,
            "--models", *missing_geom,
        ])

    missing_fixed = [m for m in args.models if not cache_has_npz(args.fixed_cache, m)]
    if missing_fixed:
        raise SystemExit(
            "fixed cache is missing for: " + ", ".join(missing_fixed) + "\n"
            f"Build it first with: {py} scripts/cache_profiles.py --ladder fixed "
            f"--cache-dir {args.fixed_cache}"
        )

    rows = []
    ladders = [
        ("fixed", args.fixed_cache),
        ("geometric", args.geometric_cache),
    ]

    for ladder, cache_dir in ladders:
        for model in args.models:
            print(f"\n===== {model}: {ladder} =====")
            grid_path = ensure_grid_tuning(
                py, model, cache_dir, args.results, args.tune_severities,
                force=args.force_tuning,
            )
            ce_path = ensure_ce_tuning(
                py, model, cache_dir, args.results, args.tune_severities,
                grid_path, args.l2, force=args.force_tuning,
            )

            for fitter, tuning_path in (("grid", grid_path), ("ce", ce_path)):
                result_tag = f"{ladder}_{fitter}"
                eval_path = run_evaluation(
                    py, model, cache_dir, tuning_path, result_tag, args.results
                )
                rows.append(row_from_eval(
                    model, ladder, fitter, cache_dir, tuning_path, eval_path
                ))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"ablation_2x2_{stamp}.json"
    csv_path = results_dir / f"ablation_2x2_{stamp}.csv"
    json_path.write_text(json.dumps(rows, indent=2))

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n=== 2x2 held-out ablation ===")
    print(f"{'model':10s} {'ladder':10s} {'fit':5s} {'mCA':>8s} {'gain':>8s} "
          f"{'NLL':>8s} {'ECE':>8s} {'a':>7s} {'b':>7s} {'g':>7s}")
    for r in rows:
        print(
            f"{r['model']:10s} {r['ladder']:10s} {r['fitter']:5s} "
            f"{100*r['mCA_fused']:7.2f}% {r['gain_pts']:+7.2f} "
            f"{r['mean_fused_nll']:8.4f} {r['mean_fused_ece']:8.4f} "
            f"{(r['alpha_global'] or 0):7.3f} {(r['beta_global'] or 0):7.3f} "
            f"{(r['gamma_global'] or 0):7.3f}"
        )
    print("saved", csv_path)
    print("saved", json_path)


if __name__ == "__main__":
    main()

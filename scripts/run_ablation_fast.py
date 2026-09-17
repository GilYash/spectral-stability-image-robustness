"""Same 2x2 orchestration as run_ablation.py, but uses tune_fusion_fast.py.

It intentionally leaves the original scripts untouched, so old published
numbers remain reproducible.  CE fitting/evaluation are unchanged; only the
legacy grid baseline is replaced by the hardware-accelerated equivalent.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Avoid Windows cp1252 failures when the project path contains Hebrew text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def latest(pattern):
    files = sorted(glob.glob(str(pattern)))
    return files[-1] if files else None


def run(cmd):
    print("\n$", " ".join(str(x) for x in cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run([str(x) for x in cmd], check=True, env=env)


def cache_has_npz(path, model):
    return any((Path(path) / model).rglob("*.npz")) if (Path(path) / model).exists() else False


def ensure_grid_tuning(py, model, cache_dir, results_dir, tune_severities,
                       *, device, config_batch, signal_chunk, tf32, force=False):
    tag = Path(cache_dir).name
    pattern = Path(results_dir) / f"tuning_fast_{model}_{tag}_*.json"
    found = latest(pattern)
    if found and not force:
        print(f"reuse grid tuning: {found}")
        return found

    cmd = [
        py, "scripts/tune_fusion_fast.py",
        "--models", model,
        "--cache-dir", cache_dir,
        "--tag", tag,
        "--out", results_dir,
        "--tune-severities", *map(str, tune_severities),
        "--device", device,
        "--config-batch", str(config_batch),
        "--signal-chunk", str(signal_chunk),
    ]
    if tf32:
        cmd.append("--tf32")
    run(cmd)
    found = latest(pattern)
    if not found:
        raise RuntimeError(f"fast grid tuner finished but no file matched {pattern}")
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
        "m_global": cfg.get("m"),
        "agreement_global": cfg.get("agreement"),
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
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"], choices=["resnet18", "deit"])
    ap.add_argument("--fixed-cache", default="cache_fixed")
    ap.add_argument("--geometric-cache", default="cache_geometric")
    ap.add_argument("--results", default="results_fast")
    ap.add_argument("--tune-severities", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--force-tuning", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--config-batch", type=int, default=64)
    ap.add_argument("--signal-chunk", type=int, default=256)
    ap.add_argument("--tf32", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    for cache_dir in (args.fixed_cache, args.geometric_cache):
        missing = [m for m in args.models if not cache_has_npz(cache_dir, m)]
        if missing:
            raise SystemExit(f"cache {cache_dir!r} is missing for: {', '.join(missing)}")

    rows = []
    for ladder, cache_dir in (("fixed", args.fixed_cache), ("geometric", args.geometric_cache)):
        for model in args.models:
            print(f"\n===== {model}: {ladder} =====")
            grid_path = ensure_grid_tuning(
                py, model, cache_dir, args.results, args.tune_severities,
                device=args.device,
                config_batch=args.config_batch,
                signal_chunk=args.signal_chunk,
                tf32=args.tf32,
                force=args.force_tuning,
            )
            ce_path = ensure_ce_tuning(
                py, model, cache_dir, args.results, args.tune_severities,
                grid_path, args.l2, force=args.force_tuning,
            )

            for fitter, tuning_path in (("grid", grid_path), ("ce", ce_path)):
                result_tag = f"{ladder}_{fitter}_fast"
                eval_path = run_evaluation(
                    py, model, cache_dir, tuning_path, result_tag, args.results
                )
                rows.append(row_from_eval(
                    model, ladder, fitter, cache_dir, tuning_path, eval_path
                ))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"ablation_2x2_fast_{stamp}.json"
    csv_path = results_dir / f"ablation_2x2_fast_{stamp}.csv"
    json_path.write_text(json.dumps(rows, indent=2))

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n=== 2x2 held-out ablation (fast grid backend) ===")
    print(f"{'model':10s} {'ladder':10s} {'fit':5s} {'mCA':>8s} {'gain':>8s} {'NLL':>8s} {'ECE':>8s}")
    for r in rows:
        print(
            f"{r['model']:10s} {r['ladder']:10s} {r['fitter']:5s} "
            f"{100*r['mCA_fused']:7.2f}% {r['gain_pts']:+7.2f} "
            f"{r['mean_fused_nll']:8.4f} {r['mean_fused_ece']:8.4f}"
        )
    print("saved", csv_path)
    print("saved", json_path)


if __name__ == "__main__":
    main()

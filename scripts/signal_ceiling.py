"""How much of the oracle gap is reachable from C_k/A_k/S_k alone?

tune_fusion.py finds the best *grid point* of alpha/beta/gamma/tau -- a coarse
search over one specific parametric family (linear combination + softmax).
This asks a sharper question: given exactly the three signals the method
computes, what's the best ANY view-reliability scorer could do, linear or
not? That separates two very different failure modes for the fused/oracle
gap: "the signals don't carry enough information" (nothing to be done without
new signals) vs. "the signals are informative but the linear-grid combination
rule wastes it" (a modelling opportunity).

Method: treat "is view k correct" as a per-view binary label, fit a
classifier on (C_k, A_k, S_k) -> P(correct), same leave-one-family-out /
tune-test split discipline as tune_fusion.py, then use the fitted P(correct)
as reliability scores and evaluate the resulting fused accuracy exactly like
evaluate.py does. Two classifiers: logistic regression (linear -- an upper
bound on any linear rule, including ours) and gradient-boosted trees
(nonlinear -- an upper bound on what these 3 signals could support at all).

    python scripts/signal_ceiling.py                      # both models, cache_fixed
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src import analysis, cache
from src.metrics import accuracy, top1
from src.splits import CORRUPTION_FAMILY, FAMILIES, leave_one_family_out, tune_test_masks


def signals_and_labels(model_name, cache_dir, corruptions, severities,
                       frac_tune, split_seed, half, m=1, agreement="all",
                       ladder_len=None):
    """Stack (C, A, S) features and per-view correctness labels over many
    conditions, restricted to `half` ('tune' or 'test') of the images.

    Returns:
        X (M, 3) float32, y (M,) bool, plus `groups` (M,) the image index
        (for reference, unused downstream) -- M = n_images * (K+1) views.
    """
    cache.CACHE_ROOT = Path(cache_dir)
    xs, ys = [], []
    for c in corruptions:
        for sev in severities:
            if not cache.has_condition(model_name, c, sev):
                continue
            d = cache.load_condition(model_name, c, sev)
            tune_mask, test_mask = tune_test_masks(d["paths"], frac_tune, split_seed)
            mask = tune_mask if half == "tune" else test_mask
            probs = d["probs"][mask].astype(np.float32)
            labels = d["labels"][mask]
            C, A, S = analysis.profile_signals(probs, m=m, standardize=True,
                                               agreement=agreement, ladder_len=ladder_len)
            correct = top1(probs) == labels[:, None]  # (n, K+1)
            xs.append(np.stack([C, A, S], axis=-1).reshape(-1, 3))
            ys.append(correct.reshape(-1))
    if not xs:
        return None, None
    return np.concatenate(xs), np.concatenate(ys)


def fused_acc_from_scores(model_name, cache_dir, corruptions, severities,
                          frac_tune, split_seed, score_fn, m=1, agreement="all",
                          ladder_len=None):
    """Mean fused accuracy on the test half of `corruptions`, using
    `score_fn(X)` -> per-view reliability score in place of the linear
    alpha*C+beta*A+gamma*S rule."""
    cache.CACHE_ROOT = Path(cache_dir)
    accs = []
    for c in corruptions:
        for sev in severities:
            if not cache.has_condition(model_name, c, sev):
                continue
            d = cache.load_condition(model_name, c, sev)
            _, test_mask = tune_test_masks(d["paths"], frac_tune, split_seed)
            probs = d["probs"][test_mask].astype(np.float32)
            labels = d["labels"][test_mask]
            C, A, S = analysis.profile_signals(probs, m=m, standardize=True,
                                               agreement=agreement, ladder_len=ladder_len)
            X = np.stack([C, A, S], axis=-1).reshape(-1, 3)
            scores = score_fn(X).reshape(C.shape)
            w = analysis.fusion_weights(scores, tau=1.0)
            fused_pred = top1(analysis.fuse(probs, w))
            accs.append(accuracy(fused_pred, labels))
    return float(np.mean(accs)) if accs else float("nan")


def main():
    """Fit and evaluate the per-family ceiling classifiers for each model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit"],
                    choices=["resnet18", "deit"])
    ap.add_argument("--cache-dir", default="cache_fixed")
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--frac-tune", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--agreement", choices=["all", "origin"], default="origin")
    ap.add_argument("--ladder-len", type=int, default=None)
    ap.add_argument("--no-gbm", action="store_true",
                    help="skip the gradient-boosted-trees ceiling (the slow part); "
                         "use once GBM has been checked on a subset and ties "
                         "logistic regression, to save time on the full run")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for model_name in args.models:
        print(f"\n=== {model_name}  ({args.cache_dir}) ===")
        per_family = {}
        for held, tune_corr, test_corr in leave_one_family_out():
            X, y = signals_and_labels(
                model_name, args.cache_dir, tune_corr, args.severities,
                args.frac_tune, args.split_seed, "tune",
                m=args.m, agreement=args.agreement, ladder_len=args.ladder_len,
            )
            if X is None:
                continue

            lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
            lr_acc = fused_acc_from_scores(
                model_name, args.cache_dir, test_corr, args.severities,
                args.frac_tune, args.split_seed,
                # decision_function (the log-odds), not predict_proba: raw
                # probabilities are compressed into [0, 1] and produce a much
                # flatter softmax than the additive alpha*C+beta*A+gamma*S
                # scores our own method uses, silently biasing this "ceiling"
                # towards equal-averaging.
                lambda X_, m=lr: m.decision_function(X_),
                m=args.m, agreement=args.agreement, ladder_len=args.ladder_len,
            )
            entry = {"logistic_acc": lr_acc}
            msg = f"  hold out {held:9s}: logistic {lr_acc*100:.1f}%"

            if not args.no_gbm:
                gb = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, subsample=0.5, random_state=0,
                ).fit(X, y)
                gb_acc = fused_acc_from_scores(
                    model_name, args.cache_dir, test_corr, args.severities,
                    args.frac_tune, args.split_seed,
                    lambda X_, m=gb: m.decision_function(X_),
                    m=args.m, agreement=args.agreement, ladder_len=args.ladder_len,
                )
                entry["gbm_acc"] = gb_acc
                msg += f"   gbm {gb_acc*100:.1f}%"

            print(msg + f"   (n_train_views={len(y)})")
            per_family[held] = entry

        report[model_name] = per_family

    path = out_dir / f"signal_ceiling_{Path(args.cache_dir).name}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()

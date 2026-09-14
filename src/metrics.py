"""Metrics over a cached condition: accuracy, calibration, oracle bound, and the
negative / positive transfer rates. All pure NumPy over 'probs (N, K+1, C)'.

The reliability signals C_k, A_k, S_k do not depend on the fusion config, so
'precompute()' builds them once per condition and 'score()' evaluates a config
against that - cheap enough to run a whole grid search per condition.
"""
from dataclasses import dataclass

import numpy as np

from src import analysis


@dataclass(frozen=True)
class FusionConfig:
    """One setting of the fusion hyper-parameters.

    Attributes:
        alpha, beta, gamma: weights on confidence, agreement, stability.
        tau: softmax temperature for the fusion weights.
        standardize: z-score each signal per image before weighting.
        m: neighbourhood radius (ladder rungs) for the stability signal.
        agreement: "all" or "origin" -- see 'analysis.profile_signals'.
    """
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    tau: float = 1.0
    standardize: bool = True
    m: int = 1
    agreement: str = "all"

    def as_dict(self):
        """Plain dict of the fields, for JSON output."""
        return dict(alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                    tau=self.tau, standardize=self.standardize, m=self.m,
                    agreement=self.agreement)

    @classmethod
    def from_dict(cls, d):
        """Rebuild from a dict, ignoring unknown keys and filling missing ones."""
        return cls(**{k: d[k] for k in
                      ("alpha", "beta", "gamma", "tau", "standardize", "m", "agreement")
                      if k in d})


def top1(probs):
    """Argmax over the last (class) axis -> predicted labels."""
    return probs.argmax(axis=-1)


def accuracy(pred, labels):
    """Fraction of 'pred' that equals 'labels' (both 1-D int arrays)."""
    return float(np.mean(pred == labels))


def ece(probs2d, labels, n_bins=15):
    """Expected Calibration Error of a set of probability vectors.

    Bins predictions by their top-class probability and averages, over bins,
    |accuracy - mean confidence| weighted by bin population.

    Args:
        probs2d: (N, C) array of probability vectors.
        labels: (N,) true labels.
        n_bins: number of equal-width confidence bins.

    Returns:
        float in [0, 1]; lower is better calibrated.
    """
    conf = probs2d.max(axis=-1)
    pred = probs2d.argmax(axis=-1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def transfer_rates(orig_pred, new_pred, labels):
    """How a new predictor moves images relative to the unfiltered view.

    Args:
        orig_pred: predictions of view 0.
        new_pred: predictions of the fused predictor.
        labels: true labels.

    Returns:
        dict with 'negative_transfer' (share of originally-correct images the
        new predictor breaks), 'positive_transfer' (share of originally-wrong
        images it fixes), and the raw counts 'n_broken', 'n_fixed'.
    """
    orig_ok = orig_pred == labels
    new_ok = new_pred == labels
    n_orig_ok = int(orig_ok.sum())
    n_orig_bad = int((~orig_ok).sum())
    broke = int((orig_ok & ~new_ok).sum())
    fixed = int((~orig_ok & new_ok).sum())
    return {
        "negative_transfer": broke / n_orig_ok if n_orig_ok else 0.0,
        "positive_transfer": fixed / n_orig_bad if n_orig_bad else 0.0,
        "n_broken": broke,
        "n_fixed": fixed,
    }


def precompute(probs_views, m=1, agreement="all", ladder_len=None):
    """Do the config-independent work for a condition once.

    Args:
        probs_views: (N, K+1, C) profile array (float16 from the cache is fine).
        m: neighbourhood radius for the stability signal.
        agreement: "all" or "origin" -- see 'analysis.profile_signals'.
        ladder_len: see 'analysis.stability_from_d'. None for a single-ladder
            profile (the spectral-only caches); set to the ladder's length for
            a combined profile (see scripts/build_combined_cache.py).

    Returns:
        dict passed to 'score()': the float32 'probs', the raw (un-standardised)
        signals 'C', 'A', 'S', the unfiltered-view probabilities 'orig', and the
        equal-weight-average predictions 'equal_pred'.
    """
    probs_views = np.asarray(probs_views, dtype=np.float32)
    C, A, S = analysis.profile_signals(probs_views, m=m, standardize=False,
                                       agreement=agreement, ladder_len=ladder_len)
    return {
        "probs": probs_views,
        "C": C, "A": A, "S": S,
        "orig": probs_views[:, 0],
        "equal_pred": top1(probs_views.mean(axis=1)),
    }


def fused_probs(pre, cfg: FusionConfig):
    """The (N, C) fused probability vectors p* for one config.

    Factored out of 'score()' so callers that need the fused *predictions*
    directly (e.g. scripts/bootstrap_ci.py) don't have to re-derive the
    reliability/softmax steps by hand.
    """
    C, A, S = pre["C"], pre["A"], pre["S"]
    if cfg.standardize:
        C, A, S = analysis.zscore(C), analysis.zscore(A), analysis.zscore(S)
    scores = analysis.reliability(C, A, S, cfg.alpha, cfg.beta, cfg.gamma)
    w = analysis.fusion_weights(scores, tau=cfg.tau)
    return analysis.fuse(pre["probs"], w)


def score(pre, labels, cfg: FusionConfig, full=True):
    """Evaluate one fusion config against a precomputed condition.

    Args:
        pre: output of 'precompute()'.
        labels: (N,) true labels.
        cfg: the config to score.
        full: if False, return just the fused accuracy (fast path for tuning);
            if True, return the full metrics row.

    Returns:
        float fused accuracy when 'full' is False, otherwise a dict with the
        fused predictor's numbers ('fused_acc', 'fused_ece', transfer rates)
        and every same-family baseline it is judged against: 'orig_acc'
        (view 0 alone), 'equal_acc' (equal-weight average), 'conf_acc'
        (confidence-only weighting), 'agree_acc' (agreement-only weighting),
        'view1_acc' (the single mildest-filtered view, no fusion),
        'oracle_acc' (per-image upper bound), and 'orig_ece'.
    """
    labels = np.asarray(labels)
    fused = fused_probs(pre, cfg)
    fused_pred = top1(fused)
    fused_acc = accuracy(fused_pred, labels)
    if not full:
        return fused_acc

    orig_pred = top1(pre["orig"])
    conf_only = analysis.fuse(
        pre["probs"], analysis.fusion_weights(pre["C"], tau=cfg.tau)
    )
    agree_only = analysis.fuse(
        pre["probs"], analysis.fusion_weights(pre["A"], tau=cfg.tau)
    )
    row = {
        "n": int(len(labels)),
        "orig_acc": accuracy(orig_pred, labels),
        "equal_acc": accuracy(pre["equal_pred"], labels),
        "conf_acc": accuracy(top1(conf_only), labels),
        "agree_acc": accuracy(top1(agree_only), labels),
        "view1_acc": accuracy(top1(pre["probs"][:, 1]), labels),
        "fused_acc": fused_acc,
        "oracle_acc": float(np.mean((top1(pre["probs"]) == labels[:, None]).any(axis=1))),
        "orig_ece": ece(pre["orig"], labels),
        "fused_ece": ece(fused, labels),
    }
    row.update(transfer_rates(orig_pred, fused_pred, labels))
    return row


def evaluate(probs_views, labels, cfg: FusionConfig):
    """'precompute' + 'score' in one call, for a quick one-off evaluation."""
    return score(precompute(probs_views, m=cfg.m, agreement=cfg.agreement),
                labels, cfg, full=True)

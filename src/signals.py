# This file contains the three signals (C_k, A_k, S_k) used in the paper.
# The signals are computed from the logits of a frozen classifier applied to each view of an image.

import torch
import torch.nn.functional as F


def view_probabilities(logits):
    # logits: (K+1, num_classes) -> row-normalized probabilities
    if logits.ndim != 2:
        raise ValueError("expected (K+1, num_classes)")
    return F.softmax(logits, dim=-1)


def confidence(probs):
    # C_k = max_c p_k(c)
    if probs.ndim != 2:
        raise ValueError("expected (K+1, num_classes)")
    return probs.max(dim=-1).values


def _kl(a, b, eps=1e-12):
    # KL(a||b) = sum a log(a/b), summed over the class axis
    a = a + eps
    b = b + eps
    return (a * (a / b).log2()).sum(dim=-1)


def jensen_shannon(p, q):
    # JSD(p, q) = 0.5 KL(p||m) + 0.5 KL(q||m),  m = 0.5(p+q)
    # log2 keeps the result in [0, 1]. p, q are (..., num_classes).
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _pairwise_jsd(probs):
    # (K+1, K+1) matrix of JSD between every pair of views
    n = probs.shape[0]
    d = torch.zeros(n, n, device=probs.device, dtype=probs.dtype)
    for i in range(n):
        # broadcast row i against all rows at once
        d[i] = jensen_shannon(probs[i].unsqueeze(0), probs)
    return d


def cross_view_agreement(probs):
    # A_k = -(1/K) sum_{j != k} JSD(p_k, p_j)   -- high when a view agrees with the rest
    n = probs.shape[0]
    d = _pairwise_jsd(probs)
    return -d.sum(dim=-1) / (n - 1)


def local_stability(probs, m=1):
    # S_k = mean -JSD to neighbors within m rungs, clipped to range. proposal fixes m=1.
    n = probs.shape[0]
    d = _pairwise_jsd(probs)
    out = torch.zeros(n, device=probs.device, dtype=probs.dtype)
    for k in range(n):
        lo, hi = max(0, k - m), min(n - 1, k + m)
        idx = [j for j in range(lo, hi + 1) if j != k]
        out[k] = -d[k, idx].mean()
    return out

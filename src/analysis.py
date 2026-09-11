"""The spectral stability profile maths: the three reliability signals and the
fusion, as pure NumPy over a whole condition at once.

The array 'probs' has shape (N, K+1, C): N images, K+1 views each (view 0 is the
unfiltered image), C classes -- exactly what 'src/cache.py' stores. This is the
single implementation of the profile maths; 'src/metrics.py' and the scripts
build everything on top of it.
"""

import numpy as np

EPS = 1e-12


def kl(a, b):
    """KL(a || b) in bits, summed over the last (class) axis; 'EPS' avoids log 0."""
    a = a + EPS
    b = b + EPS
    return np.sum(a * np.log2(a / b), axis=-1)


def jensen_shannon(p, q):
    """Jensen-Shannon divergence (base 2) between two probability arrays.

    JSD(p, q) = 0.5 KL(p || m) + 0.5 KL(q || m) with m = 0.5 (p + q). Symmetric,
    and bounded in [0, 1] for valid distributions.

    Args:
        p, q: arrays that broadcast against each other, last axis = classes.

    Returns:
        array of the broadcast shape without the class axis.
    """
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def confidence(probs):
    """C_k = max_c p_k(c): the top class probability of every view.

    Args:
        probs: (..., K+1, C) array.

    Returns:
        (..., K+1) array in [0, 1].
    """
    return probs.max(axis=-1)


def pairwise_jsd(probs):
    """Jensen-Shannon divergence between every pair of views.

    Args:
        probs: (M, K+1, C) array.

    Returns:
        (M, K+1, K+1) array, base-2, in [0, 1]; the diagonal is 0.
    """
    p = probs[:, :, None, :]                     # (M, K+1, 1, C)
    q = probs[:, None, :, :]                     # (M, 1, K+1, C)
    return jensen_shannon(p, q)


def agreement_from_d(d):
    """A_k from a precomputed pairwise-JSD matrix 'd' (M, K+1, K+1).

    A_k = -(1/K) sum_{j != k} JSD(p_k, p_j). The diagonal of 'd' is 0, so
    summing the whole row and dividing by K (= n - 1) is the same thing.
    """
    n = d.shape[-1]
    return -d.sum(axis=-1) / (n - 1)


def stability_from_d(d, m=1):
    """S_k from a precomputed pairwise-JSD matrix 'd' (M, K+1, K+1).

    S_k = - mean JSD from view k to the views within 'm' ladder rungs (k
    itself excluded).
    """
    n = d.shape[-1]
    out = np.zeros(d.shape[:-1], dtype=d.dtype)
    for k in range(n):
        lo, hi = max(0, k - m), min(n - 1, k + m)
        idx = [j for j in range(lo, hi + 1) if j != k]
        out[..., k] = -d[..., k, idx].mean(axis=-1)
    return out


def zscore(x, axis=-1, eps=1e-8):
    """Standardise 'x' to mean 0, std 1 along 'axis'.

    Applied per image to each signal so C_k, A_k, S_k (which have different
    natural ranges) become comparable before they are weighted. A near-constant
    signal (std < eps) is only recentred, not scaled.
    """
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / np.where(sd < eps, 1.0, sd)


def profile_signals(probs, m=1, standardize=True, chunk=1024):
    """Compute C_k, A_k, S_k for every image in a condition.

    Args:
        probs: (N, K+1, C) array of softmax outputs.
        m: neighbourhood radius for S_k (ladder rungs).
        standardize: if True, z-score each signal per image (see 'zscore').
        chunk: number of images processed per batch, to bound the
            (chunk, K+1, K+1, C) JSD intermediate.

    Returns:
        (C, A, S), each a (N, K+1) float32 array.
    """
    probs = np.asarray(probs, dtype=np.float32)
    n = probs.shape[0]
    C = confidence(probs)
    A = np.empty(probs.shape[:2], dtype=np.float32)
    S = np.empty(probs.shape[:2], dtype=np.float32)
    for i in range(0, n, chunk):
        d = pairwise_jsd(probs[i:i + chunk])
        A[i:i + chunk] = agreement_from_d(d)
        S[i:i + chunk] = stability_from_d(d, m=m)
    if standardize:
        C, A, S = zscore(C), zscore(A), zscore(S)
    return C, A, S


def reliability(C, A, S, alpha=1.0, beta=1.0, gamma=1.0):
    """Linear combination of the signals: r = alpha*C + beta*A + gamma*S.

    Args:
        C, A, S: (N, K+1) arrays.
        alpha, beta, gamma: weight per signal (0 drops it).

    Returns:
        (N, K+1) array of reliability scores.
    """
    return alpha * C + beta * A + gamma * S


def fusion_weights(scores, tau=1.0):
    """Row-wise softmax of 'scores / tau' over the view axis.

    Args:
        scores: (N, K+1) array.
        tau: temperature; small -> near-hard selection of one view.

    Returns:
        (N, K+1) array of weights, each row non-negative and summing to 1.
    """
    z = scores / tau
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fuse(probs, weights):
    """Per-image weighted average of the view probability vectors.

    Args:
        probs: (N, K+1, C) array.
        weights: (N, K+1) array (rows should sum to 1).

    Returns:
        (N, C) array of fused predictions p*.
    """
    probs = np.asarray(probs, dtype=np.float32)
    return np.einsum("nkc,nk->nc", probs, weights.astype(np.float32))

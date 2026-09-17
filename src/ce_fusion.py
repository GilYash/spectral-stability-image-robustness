"""Cross-entropy fitting for the non-negative reliability coefficients.

This module contains only the small numerical core used by
`scripts/tune_fusion_ce.py`.  Keeping it in `src/` makes the objective and
analytic gradient easy to unit-test independently of cache I/O.

For per-image/per-view signal vectors x_k = (C_k, A_k, S_k),

    r_k = theta^T x_k,       theta = (alpha, beta, gamma) >= 0
    w_k = softmax(r)_k
    p*(y) = sum_k w_k p_k(y)

We minimize mean negative log-likelihood plus L2 regularization.  Tau is fixed
at 1 because an overall scale in theta already controls softmax sharpness.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


DEFAULT_EPS = 1e-7


def loss_and_grad(theta, X, p_true, l2=1e-3, eps=DEFAULT_EPS):
    """Return CE objective and analytic gradient.

    Args:
        theta: shape (3,), [alpha, beta, gamma].
        X: shape (N, V, 3), standardized [C, A, S] signals.
        p_true: shape (N, V), p_k(y) for the true class in every view.
        l2: coefficient of ||theta||^2.
        eps: floor for float16 probabilities before taking logs.

    Returns:
        (loss, grad), where grad has shape (3,).
    """
    theta = np.asarray(theta, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)

    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError(f"X must have shape (N, V, 3), got {X.shape}")
    if p_true.shape != X.shape[:2]:
        raise ValueError(
            f"p_true must have shape {X.shape[:2]}, got {p_true.shape}"
        )

    log_py = np.log(np.clip(p_true, eps, 1.0))
    r = np.einsum("nvd,d->nv", X, theta)

    log_den = logsumexp(r, axis=1)
    log_num = logsumexp(r + log_py, axis=1)

    nll = np.mean(log_den - log_num)
    loss = float(nll + l2 * np.dot(theta, theta))

    # w_k = softmax(r_k)
    w = np.exp(r - log_den[:, None])
    # q_k = softmax(r_k + log p_k(y))
    #     = w_k p_k(y) / p*(y)
    q = np.exp(r + log_py - log_num[:, None])

    # d[-log p*(y)]/d theta = sum_k (w_k - q_k) x_k
    grad = np.mean(
        np.sum((w - q)[..., None] * X, axis=1),
        axis=0,
    )
    grad = grad + 2.0 * l2 * theta
    return loss, np.asarray(grad, dtype=np.float64)


def fit_nonnegative(
    X,
    p_true,
    *,
    l2=1e-3,
    eps=DEFAULT_EPS,
    starts=None,
    random_restarts=4,
    seed=0,
    maxiter=500,
):
    """Fit alpha,beta,gamma >= 0 with L-BFGS-B.

    Multiple starts are cheap because there are only three parameters.  The
    caller can pass the old grid winner as one start; deterministic random
    restarts are added automatically.

    Returns:
        (theta, best_loss, scipy_result)
    """
    X = np.asarray(X, dtype=np.float32)
    p_true = np.asarray(p_true, dtype=np.float32)

    base_starts = [] if starts is None else [np.asarray(s, dtype=float) for s in starts]
    # Always include interpretable starts, including equal reliability scores.
    base_starts.extend([
        np.zeros(3, dtype=float),
        np.ones(3, dtype=float),
    ])

    rng = np.random.default_rng(seed)
    for _ in range(random_restarts):
        base_starts.append(rng.uniform(0.0, 2.0, size=3))

    # Deduplicate after clipping to the feasible set.
    unique = []
    seen = set()
    for s in base_starts:
        s = np.maximum(np.asarray(s, dtype=float), 0.0)
        key = tuple(np.round(s, 12))
        if key not in seen:
            seen.add(key)
            unique.append(s)

    def fun(theta):
        return loss_and_grad(theta, X, p_true, l2=l2, eps=eps)

    best = None
    for x0 in unique:
        res = minimize(
            fun,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * 3,
            options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8},
        )
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError("no CE optimization starts were generated")
    return np.asarray(best.x, dtype=float), float(best.fun), best

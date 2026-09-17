import numpy as np

from src.ce_fusion import fit_nonnegative, loss_and_grad


def test_theta_zero_is_equal_weight_nll():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 5, 3)).astype(np.float32)
    p_true = rng.uniform(0.01, 0.9, size=(7, 5)).astype(np.float32)

    loss, _ = loss_and_grad(np.zeros(3), X, p_true, l2=0.0)
    expected = -np.mean(np.log(p_true.mean(axis=1)))
    assert np.allclose(loss, expected, rtol=1e-7, atol=1e-7)


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(11, 6, 3)).astype(np.float64)
    p_true = rng.uniform(0.02, 0.95, size=(11, 6)).astype(np.float64)
    theta = np.array([0.7, 0.4, 1.2], dtype=np.float64)

    loss, grad = loss_and_grad(theta, X, p_true, l2=1e-3)
    h = 1e-6
    fd = np.zeros_like(theta)
    for j in range(3):
        plus = theta.copy()
        minus = theta.copy()
        plus[j] += h
        minus[j] -= h
        lp, _ = loss_and_grad(plus, X, p_true, l2=1e-3)
        lm, _ = loss_and_grad(minus, X, p_true, l2=1e-3)
        fd[j] = (lp - lm) / (2 * h)

    assert np.isfinite(loss)
    assert np.allclose(grad, fd, rtol=2e-5, atol=2e-6)


def test_optimizer_respects_nonnegative_constraint():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 4, 3)).astype(np.float32)
    p_true = rng.uniform(0.02, 0.95, size=(20, 4)).astype(np.float32)

    theta, loss, res = fit_nonnegative(
        X, p_true, l2=1e-3, random_restarts=1, seed=0, maxiter=100
    )
    assert np.all(theta >= -1e-12)
    assert np.isfinite(loss)
    assert res.x.shape == (3,)

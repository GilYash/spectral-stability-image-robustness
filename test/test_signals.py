# Sanity checks for the reliability signals. No GPU or data needed:
#   python -m tests.test_signals

import torch
from src.signals import (
    jensen_shannon, cross_view_agreement, local_stability, view_probabilities,
)


def test_jsd_identical_is_zero():
    p = torch.tensor([[0.2, 0.3, 0.5]])
    assert torch.allclose(jensen_shannon(p, p), torch.zeros(1), atol=1e-6)


def test_jsd_disjoint_is_one():
    # with log2, JSD of two disjoint one-hots is exactly 1
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0]])
    assert torch.allclose(jensen_shannon(a, b), torch.ones(1), atol=1e-6)


def test_jsd_symmetric():
    x = torch.tensor([[0.1, 0.6, 0.3]])
    y = torch.tensor([[0.5, 0.4, 0.1]])
    assert torch.allclose(jensen_shannon(x, y), jensen_shannon(y, x))


def test_identical_stack_has_zero_signals():
    same = torch.ones(4, 3) / 3
    assert torch.allclose(cross_view_agreement(same), torch.zeros(4), atol=1e-6)
    assert torch.allclose(local_stability(same, m=1), torch.zeros(4), atol=1e-6)


def test_signal_shapes():
    probs = view_probabilities(torch.randn(6, 1000))
    assert cross_view_agreement(probs).shape == (6,)
    assert local_stability(probs, m=1).shape == (6,)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all passed")

"""Fabricate a tiny synthetic cache so the whole analysis pipeline can be
exercised end to end without ImageNet or a GPU.

The fake profiles are built so that filtered views genuinely help on some
conditions and hurt on others, which is enough to check that every script runs,
every join lines up, and no shape or index is wrong.
"""
import numpy as np
from pathlib import Path
from src.splits import ALL_CORRUPTIONS, SEVERITIES, family_of

N, V, C = 240, 6, 1000
rng = np.random.default_rng(7)
root = Path("cache")


def fake(labels, strength, helpful):
    logits = rng.normal(size=(N, V, C)).astype(np.float32) * 0.7
    boost = np.linspace(0, helpful, V) - strength * np.linspace(0, 1, V) ** 2
    logits[np.arange(N)[:, None], :, labels[:, None]] += (3.0 + boost)[None, :]
    e = np.exp(logits - logits.max(axis=2, keepdims=True))
    return (e / e.sum(axis=2, keepdims=True)).astype(np.float16)


if __name__ == "__main__":
    root.mkdir(exist_ok=True)
    labels = rng.integers(0, C, size=N)
    np.save(root / "labels.npy", labels)

    for model, scale in [("resnet18", 1.0), ("deit_tiny", 0.5)]:
        for view_kind in ("lowpass", "spatial"):
            d = root / model / ("spatial" if view_kind == "spatial" else "")
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "clean.npy", fake(labels, 2.0, 0.0))
            for c in ALL_CORRUPTIONS:
                helpful = {"noise": 1.6, "digital": 0.8, "blur": -0.4,
                           "weather": 0.2}[family_of(c)] * scale
                if view_kind == "spatial":
                    helpful *= 0.3
                for s in SEVERITIES:
                    np.save(d / f"{c}_s{s}.npy", fake(labels, 0.5 * s, helpful * s / 3))
    print(f"fake cache written: {N} images, {V} views, "
          f"{len(list(root.rglob('*.npy')))} arrays")

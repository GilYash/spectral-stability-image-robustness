"""Offline checks for every module that does not need torch or ImageNet."""
import numpy as np

from src.baselines import (agreement_only, confidence_only, equal_average,
                           headroom_report, oracle_accuracy, original)
from src.filters import (DEFAULT_CUTOFFS, apply_bank_numpy, build_bank,
                         radial_energy_profile)
from src.metrics import ece, negative_transfer, summarize
from src.signals import fuse, js_matrix, reliability_signals
from src.splits import ALL_CORRUPTIONS, image_split, leave_one_family_out

rng = np.random.default_rng(0)
fail = 0


def check(name, cond, extra=""):
    global fail
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")
    fail += (not cond)


# ---- filters -------------------------------------------------------------
bank = build_bank()
check("bank shape", bank.shape == (len(DEFAULT_CUTOFFS), 224, 113), str(bank.shape))
check("view 0 is identity mask", np.allclose(bank[0], 1.0))
check("masks in [0,1]", bank.min() >= 0 and bank.max() <= 1.0 + 1e-6)
check("masks monotonically tighter",
      all(bank[k + 1].sum() < bank[k].sum() for k in range(len(bank) - 1)))
check("DC gain preserved", all(abs(bank[k][0, 0] - 1.0) < 1e-6 for k in range(len(bank))))

x = rng.random((3, 224, 224)).astype(np.float32)
views = apply_bank_numpy(x, bank)
check("identity round-trip", np.abs(views[0] - x).max() < 1e-5,
      f"max err {np.abs(views[0] - x).max():.2e}")
check("views in [0,1]", views.min() >= 0 and views.max() <= 1)
hf = [float(np.abs(np.diff(views[k], axis=-1)).mean()) for k in range(len(bank))]
check("high-freq energy decreases with k", all(hf[i] > hf[i + 1] for i in range(len(hf) - 1)),
      " > ".join(f"{h:.4f}" for h in hf))

c, e = radial_energy_profile(x[0])
check("radial energy profile shape", c.shape == (32,) and e.shape == (32,))

# ---- signals -------------------------------------------------------------
N, V, C = 200, len(bank), 50
logits = rng.normal(size=(N, V, C)) * 2
P = np.exp(logits) / np.exp(logits).sum(axis=2, keepdims=True)

D = js_matrix(P)
check("JS symmetric", np.allclose(D, D.transpose(0, 2, 1), atol=1e-5))
check("JS zero diagonal", np.abs(np.einsum("nvv->nv", D)).max() < 1e-6)
check("JS bounded by ln2", D.max() <= np.log(2) + 1e-4, f"max {D.max():.4f}")
check("JS(p,p)=0", js_matrix(np.repeat(P[:, :1], 2, axis=1)).max() < 1e-6)

conf, agree, stab = reliability_signals(P)
check("signal shapes", conf.shape == agree.shape == stab.shape == (N, V))
check("confidence == max prob", np.allclose(conf, P.max(axis=2)))
check("agreement non-positive", agree.max() <= 1e-6)
check("stability non-positive", stab.max() <= 1e-6)

agree_o = reliability_signals(P, agreement="original")[1]
check("agreement-to-original is 0 at k=0", np.abs(agree_o[:, 0]).max() < 1e-6)

fused, w = fuse(P, conf, agree, stab, 1.0, 1.0, 1.0, 1.0)
check("weights sum to 1", np.allclose(w.sum(axis=1), 1.0))
check("fused is a distribution", np.allclose(fused.sum(axis=1), 1.0))
check("fused inside convex hull",
      (fused.max() <= P.max() + 1e-6) and (fused.min() >= P.min() - 1e-6))

f_hot, w_hot = fuse(P, conf, agree, stab, 1.0, 0.0, 0.0, 1e-3)
check("tau->0 selects argmax signal",
      (w_hot.argmax(axis=1) == conf.argmax(axis=1)).mean() > 0.99)

f_cold, _ = fuse(P, conf, agree, stab, 1.0, 1.0, 1.0, 1e6)
check("tau->inf is equal averaging", np.allclose(f_cold, equal_average(P), atol=1e-4))

# ---- metrics & oracle ----------------------------------------------------
labels = rng.integers(0, C, size=N)
base = original(P)
check("oracle >= original", oracle_accuracy(P, labels) >= (base.argmax(1) == labels).mean())
h = headroom_report(P, labels)
check("headroom non-negative", h["headroom"] >= -1e-9, f"{h['headroom']:.4f}")
check("per-view acc length", len(h["per_view_acc"]) == V)

onehot = np.eye(C)[labels]
check("ECE of perfect model ~ 0", ece(onehot, labels) < 1e-6)
check("neg transfer of identity is 0", negative_transfer(base, base, labels) == 0.0)
s = summarize(base, equal_average(P), labels)
check("summarize keys", set(s) == {"acc", "acc_base", "delta", "ece", "neg_transfer", "pos_transfer"})
check("delta consistent", abs(s["delta"] - (s["acc"] - s["acc_base"])) < 1e-12)
for probs in (confidence_only(P), agreement_only(P)):
    check("baseline is a distribution", np.allclose(probs.sum(axis=1), 1.0))

# ---- splits --------------------------------------------------------------
check("15 corruptions", len(ALL_CORRUPTIONS) == 15, str(len(ALL_CORRUPTIONS)))
folds = list(leave_one_family_out())
check("4 folds", len(folds) == 4)
check("folds are disjoint", all(not set(t) & set(e) for _, t, e in folds))
check("folds cover all", all(sorted(t + e) == sorted(ALL_CORRUPTIONS) for _, t, e in folds))
a_idx, b_idx = image_split(1000)
check("image split disjoint", len(set(a_idx) & set(b_idx)) == 0)
check("image split covers", len(a_idx) + len(b_idx) == 1000)

print(f"\n{'ALL CHECKS PASSED' if fail == 0 else str(fail) + ' CHECK(S) FAILED'}")
raise SystemExit(1 if fail else 0)

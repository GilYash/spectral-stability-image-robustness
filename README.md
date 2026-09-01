# Spectral Stability Profiles for Robust Test-Time Image Classification

Modern Computer Vision final project. Tomer Chaiat, Omer Yahav, Hadar Etzion,
Gil Yashayev, Saar Misk.

A per-image **spectral stability profile** is a frozen classifier's ordered
response to progressively low-pass filtered versions of a test image. We ask
whether that profile can be used to adaptively combine frequency views at
inference time and improve corruption robustness — with no retraining, no
corruption labels, no source data and no parameter updates.

## Core idea, and why the code is shaped this way

For a test image `x` we build `K+1` views `x_k = F^-1(F(x) * M_k)` where `M_0`
is the identity and the rest are progressively stronger smooth radial low-pass
masks. Each view goes through the same frozen classifier, giving the profile
`S(x) = (p_0, ..., p_K)`. Everything after that — the reliability signals
`C_k, A_k, S_k`, the fusion weights, every baseline and every ablation — is a
**pure function of `S(x)`**.

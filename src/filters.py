import torch


def radial_lowpass_mask(height, width, cutoff, device=None, dtype=torch.float32):
    # Gaussian low-pass in the Fourier domain. cutoff in (0, 1]: smaller = stronger
    # filtering. Frequencies are normalized so 0 sits at the center after fftshift.
    if not (0.0 < cutoff <= 1.0):
        raise ValueError("cutoff must be in (0, 1]")

    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r2 = xx ** 2 + yy ** 2
    return torch.exp(-r2 / (2.0 * cutoff ** 2))


def apply_lowpass(image, cutoff):
    # image: (C, H, W) or (B, C, H, W) in [0, 1]. Same mask on every channel.
    if image.ndim not in (3, 4):
        raise ValueError("expected (C, H, W) or (B, C, H, W)")

    h, w = image.shape[-2:]
    mask = radial_lowpass_mask(h, w, cutoff, device=image.device, dtype=image.dtype)

    spectrum = torch.fft.fft2(image, dim=(-2, -1))
    spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
    spectrum = spectrum * mask
    spectrum = torch.fft.ifftshift(spectrum, dim=(-2, -1))
    filtered = torch.fft.ifft2(spectrum, dim=(-2, -1)).real

    # inverse FFT can overshoot [0, 1] slightly
    return filtered.clamp(0.0, 1.0)


def make_spectral_views(image, cutoffs):
    # Ordered view stack. View 0 is the untouched image (M_0 == 1), the rest are
    # progressively stronger low-passes. Returns (K+1, C, H, W), or (B, K+1, C, H, W)
    # for a batch.
    if image.ndim not in (3, 4):
        raise ValueError("expected (C, H, W) or (B, C, H, W)")

    views = [image]
    for c in cutoffs:
        views.append(apply_lowpass(image, float(c)))

    return torch.stack(views, dim=0 if image.ndim == 3 else 1)


def energy_retained(image, cutoff):
    # fraction of spectral energy kept by the low-pass at this cutoff, in (0, 1]
    spectrum = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1))
    h, w = image.shape[-2:]
    mask = radial_lowpass_mask(h, w, cutoff, device=image.device, dtype=image.dtype)
    total = (spectrum.abs() ** 2).sum()
    kept = ((spectrum * mask).abs() ** 2).sum()
    return float((kept / total).item())


def cutoff_for_energy(image, target, lo=1e-3, hi=1.0, iters=40):
    # find the cutoff whose low-pass retains `target` fraction of energy, for this
    # image, by bisection. energy_retained is monotonic in cutoff so this converges.
    if not (0.0 < target <= 1.0):
        raise ValueError("target must be in (0, 1]")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if energy_retained(image, mid) < target:
            lo = mid   # keep too little -> need a larger cutoff
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cutoffs_for_energy_ladder(image, targets):
    # convenience: a cutoff per target energy fraction. Pass e.g.
    # [0.99, 0.95, 0.90, 0.75, 0.50] to get an energy-spaced ladder.
    return [cutoff_for_energy(image, float(t)) for t in targets]

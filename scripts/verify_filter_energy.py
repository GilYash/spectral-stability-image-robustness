"""Check that the low-pass ladder actually spreads the retained energy out.

For a ladder of fixed cutoffs, print the fraction of spectral energy each one
keeps and confirm it decreases monotonically, and save a plot of the curve next
to the image's power spectrum. If the spread is tiny the views barely differ
and the method has nothing to work with.

    python scripts/verify_filter_energy.py --image <path to any image>
"""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.filters import radial_lowpass_mask


def energy_retained(image, cutoff):
    """Fraction of the image's spectral energy kept by a low-pass at `cutoff`.

    A standalone copy of `src.filters.energy_retained` so this diagnostic stays
    independent of changes there.
    """
    spectrum = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1))
    h, w = image.shape[-2:]
    mask = radial_lowpass_mask(h, w, cutoff, device=image.device, dtype=image.dtype)
    total = (spectrum.abs() ** 2).sum()
    kept = ((spectrum * mask).abs() ** 2).sum()
    return float((kept / total).item())


def main():
    """Print the retained-energy ladder for one image and save the diagnostic plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="figures/filter_energy.png")
    args = parser.parse_args()

    # Load and preprocess the image - resize, center crop, and convert to tensor
    pil = Image.open(args.image).convert("RGB")
    pil = TF.center_crop(TF.resize(pil, 256), [224, 224])
    image = TF.to_tensor(pil)

    # Hyper-parameters to adjust for the low-pass filter cutoffs
    cutoffs = [0.80, 0.60, 0.40, 0.25, 0.15]

    # Compute and print the energy retained for each cutoff, checking for monotonicity
    print(f"{'cutoff':>8} {'energy':>10} {'delta':>10}")
    energies, prev, monotonic = [], 1.0, True
    for c in cutoffs:
        e = energy_retained(image, c)
        if e - prev > 1e-6:
            monotonic = False
        print(f"{c:>8.2f} {e:>10.4f} {e - prev:>10.4f}")
        energies.append(e)
        prev = e
    print("monotonic:", "yes" if monotonic else "NO")

    # Plot the energy retained across the ladder of cutoffs and visualize the power spectrum of the original image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(cutoffs, energies, "o-")
    ax1.set_xlabel("cutoff")
    ax1.set_ylabel("energy retained")
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)

    power = (torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1)).abs() ** 2).mean(0)
    ax2.imshow(torch.log1p(power).cpu(), cmap="viridis")
    ax2.axis("off")

    # Save the plot
    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()

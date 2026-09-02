# First real stability profile for one image: build the view stack, run each
# view through a frozen classifier, and look at all three signals (C_k, A_k,
# S_k) across the ladder. We filter in [0,1] space and apply ImageNet
# normalization per view.

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as TF
from PIL import Image

from src.filters import make_spectral_views, cutoffs_for_energy_ladder
from src.signals import (
    view_probabilities, confidence, cross_view_agreement, local_stability,
)
from src.models import load_resnet18, load_deit_tiny
from src.imagenet_labels import class_name, true_index_from_path

# Normalization for the models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", choices=["resnet18", "deit"], default="resnet18")
    parser.add_argument("--output", default="figures/first_profile.png")
    parser.add_argument(
        "--energies", type=float, nargs="+",
        default=[0.99, 0.95, 0.90, 0.75, 0.50],
        help="target fraction of energy retained per view (sets the ladder)",
    )
    args = parser.parse_args()

    # Set the device (Cuda, MPS, or CPU) for model inference
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("device:", device)

    # Load the model based on the specified argument
    model = (load_resnet18 if args.model == "resnet18" else load_deit_tiny)(device)[0]

    # Load and preprocess the image - resize, center crop, and convert to tensor
    pil = Image.open(args.image).convert("RGB")
    pil = TF.center_crop(TF.resize(pil, 256), [224, 224])
    image = TF.to_tensor(pil)

    # Low-pass filter cutoffs based on energy retention
    cutoffs = cutoffs_for_energy_ladder(image, args.energies)
    print("energy targets:", args.energies)
    print("solved cutoffs:", [round(c, 4) for c in cutoffs])

    # Generate spectral views of the image based on the specified cutoffs
    views = make_spectral_views(image, cutoffs=cutoffs).to(device)

    # Run the model on the generated views and compute the relevant signals
    with torch.no_grad():
        logits = model(TF.normalize(views, MEAN, STD))

    # Compute the probabilities, confidence, cross-view agreement, local stability, and predictions for each view
    probs = view_probabilities(logits)
    C = confidence(probs)
    A = cross_view_agreement(probs)
    S = local_stability(probs, m=1)
    preds = probs.argmax(dim=-1)

    # Determine the true class index and WordNet ID from the image path, if available
    true_idx, wnid = true_index_from_path(args.image)
    if true_idx is not None:
        print(f"true class: {true_idx} = {class_name(true_idx)}  ({wnid})")
    else:
        print("true class: unknown (path is not ImageNet-C style)")

    labels = ["original"] + [f"E={e:.2f}" for e in args.energies]
    print(f"{'view':>10} {'C_k':>8} {'A_k':>8} {'S_k':>8} {'pred':>6} {'':>2} {'class name'}")

    # Print the computed signals and predictions for each view
    for name, ck, ak, sk, pk in zip(
        labels, C.tolist(), A.tolist(), S.tolist(), preds.tolist()
    ):
        mark = "OK" if (true_idx is not None and pk == true_idx) else "  "
        print(f"{name:>10} {ck:>8.4f} {ak:>8.4f} {sk:>8.4f} {pk:>6d} {mark:>2} {class_name(pk)}")
    print(f"{(preds == preds[0]).sum().item()}/{len(preds)} views match the original's prediction")

    # Visualize the signals (C_k, A_k, S_k) across the ladder of views
    x = range(len(C))
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for ax, sig, name in zip(axes, [C, A, S], ["C_k", "A_k", "S_k"]):
        ax.plot(x, sig.cpu(), "o-")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()

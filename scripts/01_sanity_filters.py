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

from src.filters import make_spectral_views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to an image."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/filter_sanity.png",
        help="Where to save the visualization."
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # Load image
    # -------------------------------------------------

    pil_image = Image.open(args.image).convert("RGB")

    # For now we force 224x224 so the test resembles
    # what the classifiers will eventually receive.
    pil_image = TF.resize(pil_image, 256)
    pil_image = TF.center_crop(pil_image, [224, 224])

    image = TF.to_tensor(pil_image)

    print("Image shape:", image.shape)
    print(
        "Image range:",
        float(image.min()),
        float(image.max())
    )

    # Weak -> strong filtering.
    cutoffs = [
        0.80,
        0.60,
        0.40,
        0.25,
        0.15,
    ]

    views = make_spectral_views(
        image,
        cutoffs=cutoffs
    )

    print("Views shape:", views.shape)

    expected_shape = (
        len(cutoffs) + 1,
        3,
        224,
        224,
    )

    assert views.shape == expected_shape

    # The original view should really be identical.
    assert torch.equal(
        views[0],
        image
    )

    print("Original view preserved exactly: YES")

    # -------------------------------------------------
    # Visualize
    # -------------------------------------------------

    titles = [
        "Original",
        *[f"cutoff={c}" for c in cutoffs],
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 8)
    )

    axes = axes.flatten()

    for ax, view, title in zip(
        axes,
        views,
        titles,
    ):
        ax.imshow(
            view.permute(1, 2, 0).cpu()
        )
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
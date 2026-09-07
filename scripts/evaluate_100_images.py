import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.filters import make_spectral_views, cutoffs_for_energy_ladder
from src.models import load_resnet18, load_deit_tiny
from src.imagenet_labels import true_index_from_path

NUM_IMAGES = 100
ENERGIES_CUTOFFS = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def collect_images(root):
    root = Path(root)

    extensions = {".jpg", ".jpeg", ".png", ".JPEG"}

    images = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    return images


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/imagenet-c",
        help="root directory containing ImageNet-C images",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=NUM_IMAGES,
    )

    parser.add_argument(
        "--model",
        choices=["resnet18", "deit"],
        default="resnet18",
    )

    parser.add_argument(
        "--energies",
        type=float,
        nargs="+",
        default=ENERGIES_CUTOFFS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    # -------------------------------------------------------
    # Device
    # -------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("device:", device)

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------

    model = (
        load_resnet18
        if args.model == "resnet18"
        else load_deit_tiny
    )(device)[0]

    # -------------------------------------------------------
    # Collect and randomly choose images
    # -------------------------------------------------------

    all_images = collect_images(args.data)

    print(f"found {len(all_images)} images")

    random.seed(args.seed)

    if len(all_images) < args.num_images:
        raise ValueError(
            f"requested {args.num_images} images, "
            f"but only found {len(all_images)}"
        )

    selected_images = random.sample(
        all_images,
        args.num_images,
    )

    # -------------------------------------------------------
    # Statistics
    # -------------------------------------------------------

    labels = ["original"] + [
        f"E={e:.2f}" for e in args.energies
    ]

    correct_counts = [0] * len(labels)

    # We only compare FILTERED views to the original
    match_original_counts = [0] * len(args.energies)

    # Filtered view is correct while the original prediction is wrong
    rescued_counts = [0] * len(args.energies)

    valid_images = 0

    # -------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------

    for i, image_path in enumerate(selected_images, start=1):

        true_idx, _ = true_index_from_path(str(image_path))

        if true_idx is None:
            print(
                f"skipping {image_path}: "
                "could not determine true class"
            )
            continue

        pil = Image.open(image_path).convert("RGB")

        pil = TF.center_crop(
            TF.resize(pil, 256),
            [224, 224],
        )

        image = TF.to_tensor(pil)

        # Per-image energy ladder
        cutoffs = cutoffs_for_energy_ladder(
            image,
            args.energies,
        )

        views = make_spectral_views(
            image,
            cutoffs,
        ).to(device)

        # Normalize all views
        views_normalized = TF.normalize(
            views,
            MEAN,
            STD,
        )

        with torch.no_grad():
            logits = model(views_normalized)

        preds = logits.argmax(dim=-1)

        # ---------------------------------------------------
        # Correct predictions
        # ---------------------------------------------------

        for k, pred in enumerate(preds):
            if pred.item() == true_idx:
                correct_counts[k] += 1

        # ---------------------------------------------------
        # Agreement with original
        # ---------------------------------------------------

        original_pred = preds[0]
        original_correct = original_pred.item() == true_idx

        for k in range(1, len(preds)):
            filtered_pred = preds[k]

            # Same prediction as the original
            if filtered_pred == original_pred:
                match_original_counts[k - 1] += 1

            # Filtered view rescued an image that the original got wrong
            if not original_correct and filtered_pred.item() == true_idx:
                rescued_counts[k - 1] += 1

        valid_images += 1

        if i % 10 == 0:
            print(
                f"processed {i}/{args.num_images}"
            )

    # -------------------------------------------------------
    # Results
    # -------------------------------------------------------

    print()
    print("=" * 75)
    print("RESULTS")
    print("=" * 75)

    print(
        f"{'view':>10} "
        f"{'correct':>12} "
        f"{'accuracy':>12} "
        f"{'same as original':>18} "
        f"{'rescued':>14}"
    )

    print("-" * 75)

    for k, label in enumerate(labels):

        correct = correct_counts[k]

        accuracy = (
            correct / valid_images
            if valid_images > 0
            else 0
        )

        if k == 0:
            same_text = "-"
            rescued_text = "-"
        else:
            same = match_original_counts[k - 1]
            rescued = rescued_counts[k - 1]

            same_text = (
                f"{same}/{valid_images} "
                f"({100 * same / valid_images:.1f}%)"
            )

            rescued_text = f"{rescued}/{valid_images}"

        print(
            f"{label:>10} "
            f"{correct:>5}/{valid_images:<6} "
            f"{100 * accuracy:>10.1f}% "
            f"{same_text:>18} "
            f"{rescued_text:>14}"
        )

    print("=" * 75)


if __name__ == "__main__":
    main()
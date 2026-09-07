import argparse
import os
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


ENERGIES_CUTOFFS = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_subset(subset_path, data_root):
    subset_path = Path(subset_path)
    data_root = Path(data_root)

    with open(subset_path, "r", encoding="utf-8") as f:
        relative_paths = [
            line.strip()
            for line in f
            if line.strip()
        ]

    image_paths = [
        data_root / relative_path
        for relative_path in relative_paths
    ]

    missing = [
        path for path in image_paths
        if not path.exists()
    ]

    if missing:
        print("Missing images:")
        for path in missing:
            print(" ", path)

        raise FileNotFoundError(
            f"{len(missing)} selected images were not found"
        )

    return image_paths


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/imagenet-c/gaussian_noise/1",
        help="root directory for the selected corruption/severity",
    )

    parser.add_argument(
        "--subset",
        default="data/clean_subset.txt",
        help="file containing selected relative image paths",
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
    # Load selected subset
    # -------------------------------------------------------

    selected_images = load_subset(
        args.subset,
        args.data,
    )

    num_images = len(selected_images)

    print(f"loaded {num_images} selected images")

    labels = ["original"] + [
        f"E={e:.2f}" for e in args.energies
    ]

    # -------------------------------------------------------
    # Statistics
    # -------------------------------------------------------

    correct_counts = [0] * len(labels)

    # Filtered prediction equals original prediction
    match_original_counts = [0] * len(args.energies)

    # Original wrong -> filtered correct
    rescued_counts = [0] * len(args.energies)

    # Original correct -> filtered wrong
    broken_counts = [0] * len(args.energies)

    valid_images = 0
    # Number of images for which at least one view
    # (original or filtered) predicts the correct class
    oracle_correct = 0

    # Specifically: original was wrong, but at least one filtered view was correct
    oracle_rescued = 0

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

        # Find per-image Gaussian cutoffs that retain
        # the requested fractions of spectral energy
        cutoffs = cutoffs_for_energy_ladder(
            image,
            args.energies,
        )

        views = make_spectral_views(
            image,
            cutoffs,
        ).to(device)

        views_normalized = TF.normalize(
            views,
            MEAN,
            STD,
        )

        with torch.no_grad():
            logits = model(views_normalized)

        preds = logits.argmax(dim=-1)

        # ---------------------------------------------------
        # Accuracy
        # ---------------------------------------------------

        for k, pred in enumerate(preds):
            if pred.item() == true_idx:
                correct_counts[k] += 1

        original_pred = preds[0].item()
        original_correct = original_pred == true_idx

        # Oracle: was ANY view correct?
        any_view_correct = (preds == true_idx).any().item()

        if any_view_correct:
            oracle_correct += 1

        # Headroom created specifically by the filtered views
        if not original_correct and any_view_correct:
            oracle_rescued += 1

        # ---------------------------------------------------
        # Compare filtered views with original
        # ---------------------------------------------------

        for k in range(1, len(preds)):
            filtered_pred = preds[k].item()

            if filtered_pred == original_pred:
                match_original_counts[k - 1] += 1

            # Original failed, filtering rescued it
            if (
                not original_correct
                and filtered_pred == true_idx
            ):
                rescued_counts[k - 1] += 1

            # Original was correct, filtering broke it
            if (
                original_correct
                and filtered_pred != true_idx
            ):
                broken_counts[k - 1] += 1

        valid_images += 1

        if i % 10 == 0 or i == num_images:
            print(
                f"processed {i}/{num_images}"
            )

    # -------------------------------------------------------
    # Results
    # -------------------------------------------------------

    print()
    print("=" * 105)
    print("CLEAN SUBSET RESULTS")
    print("=" * 105)

    print(
        f"{'view':>10} "
        f"{'correct':>12} "
        f"{'accuracy':>10} "
        f"{'same as original':>20} "
        f"{'rescued':>12} "
        f"{'broken':>12}"
    )

    print("-" * 105)

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
            broken_text = "-"
        else:
            same = match_original_counts[k - 1]
            rescued = rescued_counts[k - 1]
            broken = broken_counts[k - 1]

            same_text = (
                f"{same}/{valid_images} "
                f"({100 * same / valid_images:.1f}%)"
            )

            rescued_text = f"{rescued}/{valid_images}"
            broken_text = f"{broken}/{valid_images}"

        print(
            f"{label:>10} "
            f"{correct:>5}/{valid_images:<6} "
            f"{100 * accuracy:>8.1f}% "
            f"{same_text:>20} "
            f"{rescued_text:>12} "
            f"{broken_text:>12}"
        )

    print("=" * 105)
    print()
    print("ORACLE ANALYSIS")
    print("-" * 50)

    original_correct = correct_counts[0]
    original_wrong = valid_images - original_correct

    print(
        f"Original accuracy: "
        f"{original_correct}/{valid_images} "
        f"({100 * original_correct / valid_images:.1f}%)"
    )

    print(
        f"Oracle accuracy:   "
        f"{oracle_correct}/{valid_images} "
        f"({100 * oracle_correct / valid_images:.1f}%)"
    )

    print(
        f"Oracle rescued:    "
        f"{oracle_rescued}/{original_wrong} "
        f"({100 * oracle_rescued / original_wrong:.1f}% of original errors)"
    )

    print(
        f"Oracle gain:       "
        f"+{oracle_correct - original_correct}/{original_wrong} "
        f"({100 * (oracle_correct - original_correct) / original_wrong:.1f}% of original errors recovered)"
    )

    print(
        f"Accuracy gain:     "
        f"+{100 * (oracle_correct - original_correct) / valid_images:.1f} percentage points"
    )


if __name__ == "__main__":
    main()
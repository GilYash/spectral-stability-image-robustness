import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from PIL import Image

from src.imagenet_labels import class_name, true_index_from_path


def collect_classes(root):
    root = Path(root)
    classes = {}

    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue

        images = [
            p for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        if images:
            classes[class_dir.name] = images

    return classes


def choose_images(images, title):
    """
    Display up to 10 images.

    Click an image to select/deselect it.
    Press Enter to confirm the current selection.
    """

    selected = set()

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()

    def refresh():
        for i, ax in enumerate(axes):
            if i >= len(images):
                ax.axis("off")
                continue

            if i in selected:
                # Selected image
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(5)
            else:
                for spine in ax.spines.values():
                    spine.set_visible(False)

            ax.set_title(
                f"{i} {'✓' if i in selected else ''}",
                fontsize=13,
            )

        fig.canvas.draw_idle()

    for i, (ax, path) in enumerate(zip(axes, images)):
        image = Image.open(path).convert("RGB")

        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(str(i), fontsize=13)

    for ax in axes[len(images):]:
        ax.axis("off")

    fig.suptitle(
        title
        + "\nClick images to select/deselect | Press ENTER to confirm",
        fontsize=16,
    )

    def on_click(event):
        if event.inaxes not in axes:
            return

        index = list(axes).index(event.inaxes)

        if index >= len(images):
            return

        if index in selected:
            selected.remove(index)
        else:
            selected.add(index)

        refresh()

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    plt.show()

    return [images[i] for i in sorted(selected)]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/imagenet-c/gaussian_noise/1",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--candidates-per-class",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        default="data/clean_subset.txt",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    classes = collect_classes(args.data)

    print(f"Found {len(classes)} classes")

    if len(classes) < args.num_classes:
        raise ValueError(
            f"Requested {args.num_classes} classes, "
            f"but only found {len(classes)}."
        )

    selected_wnids = random.sample(
        list(classes.keys()),
        args.num_classes,
    )

    final_selection = []

    for class_number, wnid in enumerate(selected_wnids, start=1):

        class_images = classes[wnid]

        if len(class_images) < args.candidates_per_class:
            candidates = class_images
        else:
            candidates = random.sample(
                class_images,
                args.candidates_per_class,
            )

        true_idx, _ = true_index_from_path(str(candidates[0]))

        readable_name = (
            class_name(true_idx)
            if true_idx is not None
            else wnid
        )

        print()
        print(
            f"Class {class_number}/{args.num_classes}: "
            f"{readable_name} ({wnid})"
        )

        chosen = choose_images(
            candidates,
            title=(
                f"Class {class_number}/{args.num_classes}: "
                f"{readable_name} ({wnid})"
            ),
        )

        print(f"Selected {len(chosen)} images")

        for path in chosen:
            # Store only class/image, not corruption/severity
            relative_path = f"{wnid}/{path.name}"
            final_selection.append(relative_path)

    # -------------------------------------------------------
    # Save selected images
    # -------------------------------------------------------

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for path in final_selection:
            f.write(path + "\n")

    print()
    print("=" * 60)
    print("SELECTION COMPLETE")
    print("=" * 60)
    print(f"Selected images: {len(final_selection)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
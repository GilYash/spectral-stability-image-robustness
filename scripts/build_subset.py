"""Pick the fixed 100-class subset used for the whole study.

ImageNet class index i == position of the i-th wnid in lexicographic order
(the torchvision / HF imagenet-1k convention). We read that 1000-wnid list
from an existing ImageNet-C severity folder, draw 100 of them uniformly at
random with a fixed seed, and write the wnid -> index map to
data/subset_classes.json. Everything downstream keys off that file.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sorted_wnids(reference_dir):
    """Return the 1000 wnid folder names in `reference_dir`, lexicographically.

    Position in this list == ImageNet class index. Exits with an error if the
    directory does not hold exactly 1000 wnid folders, since a partial list
    would give wrong indices.
    """
    import re
    wnid_re = re.compile(r"n\d{8}")
    wnids = sorted(d for d in os.listdir(reference_dir) if wnid_re.fullmatch(d))
    if len(wnids) != 1000:
        raise SystemExit(
            f"{reference_dir} has {len(wnids)} wnid folders, need all 1000 to "
            f"fix the index mapping. Point --reference-dir at a complete "
            f"ImageNet-C severity folder."
        )
    return wnids


def main():
    """Parse args, draw the subset, write data/subset_classes.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-dir", default="data/imagenet-c/gaussian_noise/1",
                    help="any folder that holds all 1000 wnid subfolders")
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/subset_classes.json")
    args = ap.parse_args()

    wnids = sorted_wnids(args.reference_dir)

    rng = random.Random(args.seed)
    picked_idx = sorted(rng.sample(range(1000), args.num_classes))
    subset = {wnids[i]: i for i in picked_idx}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"seed": args.seed, "num_classes": args.num_classes,
             "wnid_to_index": subset},
            f, indent=2,
        )
    print(f"wrote {out} : {len(subset)} classes, seed {args.seed}")
    print("first few:", list(subset.items())[:5])


if __name__ == "__main__":
    main()

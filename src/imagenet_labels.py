"""Turn an image path into its true ImageNet class index and name.

ImageNet class index i is, by the torchvision / HF convention, the position of
the i-th WordNet id (wnid, e.g. 'n01440764') in lexicographic order. Directory
names in ImageNet and ImageNet-C are wnids, so the label of an image is
determined by its parent folder name.
"""
import json
import os
import re
from pathlib import Path

from torchvision.models import ResNet18_Weights

#: Human-readable class names indexed 0..999 (ships with the weights metadata).
CATEGORIES = ResNet18_Weights.DEFAULT.meta["categories"]

WNID_RE = re.compile(r"n\d{8}")

# Authoritative wnid -> index map for the fixed study subset. When only the ~100
# subset classes are on disk, "sorted position among the folders present"
# ('wnid_to_index_map') would number them 0..99, so this file is consulted first.
SUBSET_FILE = Path(__file__).resolve().parent.parent / "data" / "subset_classes.json"
subset_map = None
if SUBSET_FILE.exists():
    subset_map = json.loads(SUBSET_FILE.read_text())["wnid_to_index"]


def class_name(index):
    """Human-readable name for an ImageNet class index (0..999)."""
    return CATEGORIES[index]


def wnid_from_path(path):
    """Extract the WordNet id from a path, or None if it contains none.

    Matches the first 'nXXXXXXXX' component, e.g. in
    '.../gaussian_noise/3/n01440764/ILSVRC2012_val_00000293.JPEG'.
    """
    m = WNID_RE.search(str(path))
    return m.group(0) if m else None


warned_dirs = set()
map_cache = {}


def wnid_to_index_map(severity_dir):
    """Map every wnid folder in 'severity_dir' to its ImageNet index.

    The index is the folder's position in lexicographic order, which is correct
    only when all 1000 classes are present (prints a warning otherwise). The
    result is cached per directory because listing 1000 folders per image would
    be slow. Used as the fallback when there is no subset map.

    Args:
        severity_dir: a directory holding wnid subfolders (an ImageNet-C
            '<corruption>/<severity>' directory, or an ImageNet split root).

    Returns:
        dict mapping wnid string -> int index.
    """
    key = str(severity_dir)
    if key in map_cache:
        return map_cache[key]
    wnids = sorted(d for d in os.listdir(severity_dir) if WNID_RE.fullmatch(d))
    if len(wnids) != 1000 and key not in warned_dirs:
        warned_dirs.add(key)
        print(f"warning: {severity_dir} has {len(wnids)} wnid folders, not 1000 "
              f"-- label indices may be wrong until extraction is complete")
    mapping = {w: i for i, w in enumerate(wnids)}
    map_cache[key] = mapping
    return mapping


def true_index_from_path(path):
    """Resolve an image path to (imagenet_index, wnid).

    Prefers 'data/subset_classes.json' (correct even when only 100 class folders
    exist); falls back to sorted-position within the image's severity directory
    when the full 1000-class tree is present.

    Args:
        path: path to an ImageNet or ImageNet-C image.

    Returns:
        (index, wnid), or (None, None) if the path contains no wnid.
    """
    path = Path(path)
    wnid = wnid_from_path(path)
    if wnid is None:
        return None, None
    if subset_map is not None and wnid in subset_map:
        return subset_map[wnid], wnid
    return wnid_to_index_map(path.parent.parent).get(wnid), wnid

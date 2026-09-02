import os
import re
from pathlib import Path

from torchvision.models import ResNet18_Weights

# human-readable class names, indexed 0..999 (ships with weights meta, no download)
CATEGORIES = ResNet18_Weights.DEFAULT.meta["categories"]

_WNID_RE = re.compile(r"n\d{8}")


def class_name(index):
    return CATEGORIES[index]


def wnid_from_path(path):
    # ImageNet-C paths look like .../<corruption>/<severity>/<wnid>/<img>.JPEG
    m = _WNID_RE.search(str(path))
    return m.group(0) if m else None


def wnid_to_index_map(severity_dir):
    # class index == position of the wnid in lexicographically sorted order.
    # every severity dir holds all 1000 wnid subfolders, so read them from there.
    # if fewer than 1000 are present the indices won't match the model, so warn.
    wnids = sorted(d for d in os.listdir(severity_dir) if _WNID_RE.fullmatch(d))
    if len(wnids) != 1000:
        print(f"warning: {severity_dir} has {len(wnids)} wnid folders, not 1000 "
              f"-- label indices may be wrong until extraction is complete")
    return {w: i for i, w in enumerate(wnids)}


def true_index_from_path(path):
    # parse the wnid and map it using the severity dir (two levels up). Returns
    # (index, wnid), or (None, None) if the path isn't an ImageNet-C image path.
    path = Path(path)
    wnid = wnid_from_path(path)
    if wnid is None:
        return None, None
    return wnid_to_index_map(path.parent.parent).get(wnid), wnid

"""PASCAL VOC preprocessing helpers used by the LLaVA-NeXt caption generator.

Loads the HuggingFace mirror `nateraw/pascal-voc-2012` and converts a
ground-truth segmentation mask into the `class_prompt` string that
`gen_data.py` and `src/datasets/voc.py::get_indices` expect.
"""

import numpy as np
from datasets import load_dataset

from src.datasets.voc import COMPOUND_NOUNS
from src.datasets.voc import classes as VOC_CLASSES
from src.datasets.voc import palette as VOC_PALETTE_FLAT

# VOC class id -> phrase used inside captions/class_prompt. Compound classes
# are emitted as their space-separated form (e.g. "dining table") so the
# LLaVA caption still reads naturally and `handle_compound_nouns` in
# src/datasets/voc.py can re-collapse them at indexing time.
_COMPOUND_PHRASE = {k: v[0] for k, v in COMPOUND_NOUNS.items()}

# Reverse-lookup RGB tuple -> class id. The HF mirror `nateraw/pascal-voc-2012`
# returns masks as RGB images rather than palette-indexed PIL images, so we
# need this to recover class ids. (224, 224, 192) is the standard VOC ignore
# color used along object boundaries.
_RGB_TO_CLASS = {
    tuple(VOC_PALETTE_FLAT[3 * i : 3 * i + 3]): i for i in range(len(VOC_CLASSES))
}
_RGB_TO_CLASS[(224, 224, 192)] = 255


def class_phrase(class_id: int) -> str:
    name = VOC_CLASSES[class_id]
    return _COMPOUND_PHRASE.get(name, name)


def load_voc_split(split: str = "train"):
    """Return the requested PASCAL VOC split from the HF mirror."""
    return load_dataset("nateraw/pascal-voc-2012", split=split)


def mask_to_class_ids(mask) -> list:
    """Extract foreground class ids from a VOC mask (RGB or palette-mode).

    Skips background (0) and the 255 ignore label. Result is sorted ascending
    for determinism.
    """
    arr = np.array(mask)
    ids = set()
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3].reshape(-1, 3)
        unique_rgb = np.unique(rgb, axis=0)
        for triple in unique_rgb:
            cid = _RGB_TO_CLASS.get(tuple(int(x) for x in triple))
            if cid is not None:
                ids.add(cid)
    else:
        for v in np.unique(arr):
            ids.add(int(v))
    ids.discard(0)
    ids.discard(255)
    return sorted(c for c in ids if 0 <= c < len(VOC_CLASSES))


def derive_class_prompt(mask) -> str:
    """Build a `class_prompt` string from a VOC segmentation mask.

    Returns a space-separated list of class phrases present in the mask
    (excluding background and the ignore label), e.g. ``"person dining table"``.
    Returns ``""`` if the mask has no foreground class — caller should skip
    such samples.
    """
    return " ".join(class_phrase(c) for c in mask_to_class_ids(mask))


if __name__ == "__main__":
    ds = load_voc_split("train")
    print(f"train: {len(ds)} samples; fields: {list(ds[0].keys())}")
    sample = ds[0]
    print(f"image: {sample['image'].size}, mask: {sample['mask'].size}")
    print(f"class_prompt: {derive_class_prompt(sample['mask'])!r}")

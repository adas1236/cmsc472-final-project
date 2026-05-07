"""DenseCRF post-processing for masks produced by ``gen_data.py``.

Reads ``{work_dir}/image/*.jpg`` + ``{work_dir}/mask/*.png`` and writes refined
masks to ``{work_dir}/{out_subdir}/`` using a fully-connected CRF
(Krähenbühl & Koltun 2011). Hard labels from the input mask are turned into
soft unaries via label smoothing (``gt_prob``); the ``255`` ignore band gets
a uniform prior so the CRF resolves it from image evidence alone.
"""

import argparse
import multiprocessing as mp
import os
import os.path as osp
from functools import partial
from glob import glob

import numpy as np
import pydensecrf.densecrf as dcrf
from PIL import Image
from pydensecrf.utils import unary_from_labels
from tqdm import tqdm

from src.seed_utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Apply DenseCRF to Dataset Diffusion masks.")
    parser.add_argument("--work-dir", required=True, help="dir containing image/ and mask/ subdirs")
    parser.add_argument("--out-subdir", default="mask_crf", help="output dir name under --work-dir")
    parser.add_argument("--data-type", choices=["voc", "coco"], default="voc")
    parser.add_argument("--gt-prob", type=float, default=0.7,
                        help="probability assigned to the labelled class in the smoothed unary")
    parser.add_argument("--gaussian-sxy", type=float, default=3.0)
    parser.add_argument("--gaussian-compat", type=float, default=3.0)
    parser.add_argument("--bilateral-sxy", type=float, default=80.0)
    parser.add_argument("--bilateral-srgb", type=float, default=13.0)
    parser.add_argument("--bilateral-compat", type=float, default=10.0)
    parser.add_argument("--n-iters", type=int, default=5)
    parser.add_argument("--keep-uncertain-threshold", type=float, default=0.5,
                        help="pixels with CRF posterior below this become 255 (ignore). "
                             "Set negative to disable.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true",
                        help="reprocess images even if output mask already exists")
    return parser.parse_args()


def get_palette(data_type):
    if data_type == "voc":
        from src.datasets.voc import palette
    elif data_type == "coco":
        from src.datasets.coco import palette
    else:
        raise ValueError(f"unknown data_type: {data_type}")
    return palette


def refine_one(mask_path, image_dir, out_dir, palette, cfg):
    base = osp.splitext(osp.basename(mask_path))[0]
    out_path = osp.join(out_dir, f"{base}.png")
    if not cfg["overwrite"] and osp.exists(out_path):
        return out_path, "skip"

    image_path = osp.join(image_dir, f"{base}.jpg")
    if not osp.exists(image_path):
        return out_path, f"missing_image:{image_path}"

    image = np.ascontiguousarray(np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8))
    mask = np.array(Image.open(mask_path).convert("P"), dtype=np.uint8)
    h, w = mask.shape

    present = sorted(int(v) for v in np.unique(mask) if v != 255)
    if len(present) <= 1:
        # Nothing to refine — copy through unchanged so downstream tooling sees a complete dir.
        out = Image.fromarray(mask, mode="P")
        out.putpalette(palette)
        out.save(out_path)
        return out_path, "passthrough"

    K = len(present)
    # Compact remap: 255 -> 0 (unsure), present[i] -> i + 1, in the [1..K] range expected
    # by ``unary_from_labels(zero_unsure=True)``.
    compact = np.zeros((h, w), dtype=np.int32)
    for i, c in enumerate(present):
        compact[mask == c] = i + 1

    unary = unary_from_labels(compact, K, gt_prob=cfg["gt_prob"], zero_unsure=True)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(w, h, K)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=cfg["gaussian_sxy"], compat=cfg["gaussian_compat"])
    d.addPairwiseBilateral(
        sxy=cfg["bilateral_sxy"],
        srgb=cfg["bilateral_srgb"],
        rgbim=image,
        compat=cfg["bilateral_compat"],
    )
    Q = np.array(d.inference(cfg["n_iters"]))  # (K, H*W)
    preds = Q.argmax(axis=0).reshape(h, w)
    max_probs = Q.max(axis=0).reshape(h, w)

    present_arr = np.array(present, dtype=np.uint8)
    refined = present_arr[preds]
    if cfg["keep_uncertain_threshold"] >= 0:
        refined[max_probs < cfg["keep_uncertain_threshold"]] = 255

    out = Image.fromarray(refined, mode="P")
    out.putpalette(palette)
    out.save(out_path)
    return out_path, "ok"


def main():
    args = parse_args()
    seed_everything(args.seed)
    image_dir = osp.join(args.work_dir, "image")
    mask_dir = osp.join(args.work_dir, "mask")
    out_dir = osp.join(args.work_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    if not osp.isdir(image_dir) or not osp.isdir(mask_dir):
        raise SystemExit(f"expected {image_dir} and {mask_dir} to exist; run gen_data.py first")

    mask_paths = sorted(glob(osp.join(mask_dir, "*.png")))
    start = max(0, args.start)
    end = len(mask_paths) if args.end == -1 else min(len(mask_paths), args.end)
    mask_paths = mask_paths[start:end]
    if not mask_paths:
        print("nothing to process")
        return

    palette = get_palette(args.data_type)
    cfg = {
        "gt_prob": args.gt_prob,
        "gaussian_sxy": args.gaussian_sxy,
        "gaussian_compat": args.gaussian_compat,
        "bilateral_sxy": args.bilateral_sxy,
        "bilateral_srgb": args.bilateral_srgb,
        "bilateral_compat": args.bilateral_compat,
        "n_iters": args.n_iters,
        "keep_uncertain_threshold": args.keep_uncertain_threshold,
        "overwrite": args.overwrite,
    }

    worker = partial(refine_one, image_dir=image_dir, out_dir=out_dir, palette=palette, cfg=cfg)

    counts = {}
    if args.num_workers <= 1:
        for p in tqdm(mask_paths, desc="dense_crf"):
            _, status = worker(p)
            counts[status] = counts.get(status, 0) + 1
    else:
        with mp.get_context("spawn").Pool(args.num_workers) as pool:
            for _, status in tqdm(
                pool.imap_unordered(worker, mask_paths, chunksize=4),
                total=len(mask_paths),
                desc="dense_crf",
            ):
                counts[status] = counts.get(status, 0) + 1

    print("done:", counts)


if __name__ == "__main__":
    main()

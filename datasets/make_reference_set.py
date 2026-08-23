#!/usr/bin/env python3
"""
Materialise a reference image folder (and optional clean-fid custom statistics)
for CIFAR-10 / ImageNet / ImageNet-LT.

FID, MMD and t-SNE all need real images that went through *exactly* the same
preprocessing as the training data.  This script writes those images as PNGs so
they can be consumed by ``cleanfid`` and by ``evaluation/evaluate_gfm.py``, and
registers a named clean-fid custom statistic so ``train.py`` can report FID
during training via ``--cleanfid_dataset_name``.

Examples
--------
CIFAR-10 reference at the native 32x32 evaluation resolution:

    python datasets/make_reference_set.py --dataset cifar10 \
        --data_root ./data/cifar10 --out_dir ./data/reference/cifar10_32 \
        --source_image_size 256 --eval_image_size 32 \
        --num_images 50000 --cleanfid_name cifar10_gfm_32

ImageNet-LT reference at 256x256:

    python datasets/make_reference_set.py --dataset imagenet-lt \
        --data_root /data/imagenet --out_dir /data/reference/imagenet_lt_256 \
        --source_image_size 256 --eval_image_size 256 \
        --num_images 50000 --cleanfid_name imagenet_lt_256
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.gfm_image_datasets import build_image_dataset


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   choices=["cifar10", "imagenet", "imagenet-lt"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--imagenet_subdir", default="train")
    p.add_argument("--imagenet_lt_split_file", default=None)
    p.add_argument("--no_pareto_fallback", action="store_true",
                   help="Fail instead of reconstructing an LT split")
    p.add_argument("--pareto_seed", type=int, default=0)

    p.add_argument("--source_image_size", type=int, default=256,
                   help="Resolution the model was trained at (must match the "
                        "encoder pass so the crop/resize pipeline is identical)")
    p.add_argument("--eval_image_size", type=int, default=256,
                   help="Resolution the metrics are computed at. Use 32 for the "
                        "standard CIFAR-10 protocol.")
    p.add_argument("--num_images", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--cleanfid_name", default=None,
                   help="If set, register clean-fid custom stats under this name")
    p.add_argument("--cleanfid_mode", default="clean",
                   choices=["clean", "legacy_pytorch", "legacy_tensorflow"])
    p.add_argument("--device", default="cuda:0",
                   help="Device for the clean-fid Inception pass. Without this "
                        "clean-fid silently defaults to cuda:0, which is the "
                        "wrong GPU on a multi-GPU box and fails on a CPU-only "
                        "one. Falls back to cpu when CUDA is unavailable.")
    p.add_argument("--stats_batch_size", type=int, default=64)
    p.add_argument("--reuse_stats", action="store_true",
                   help="Keep an existing clean-fid stats file instead of "
                        "rebuilding it. Only safe when the reference images "
                        "have not changed; by default the stats are rebuilt so "
                        "they always describe the images just written.")
    return p.parse_args(argv)


def to_uint8(x):
    """[-1, 1] float tensor -> uint8 HWC numpy."""
    x = ((x.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return x.permute(1, 2, 0).cpu().numpy()


def cleanfid_stats_exist(name, mode="clean"):
    """Does clean-fid already hold custom stats under this name?

    clean-fid has no public predicate in every version, so ask its own helper
    when it exists and fall back to the file layout make_custom_stats uses:
    ``<pkg>/stats/<name>_<mode>[_<model>]_custom_na.npz``.
    """
    try:
        from cleanfid.fid import test_stats_exists
        return bool(test_stats_exists(name, mode))
    except Exception:
        pass
    try:
        import glob
        import cleanfid
        folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
        return bool(glob.glob(os.path.join(folder, f"{name}_{mode}*custom_na.npz")))
    except Exception:
        return False


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = build_image_dataset(
        args.dataset, args.data_root, split=args.split,
        image_size=args.source_image_size, use_horizontal_flips=False,
        return_label=True,
        imagenet_subdir=args.imagenet_subdir,
        split_file=args.imagenet_lt_split_file,
        allow_pareto_fallback=not args.no_pareto_fallback,
        pareto_seed=args.pareto_seed,
    )
    print(f"[data] {args.dataset}/{args.split}: {len(ds)} images available")

    n = min(args.num_images, len(ds))
    rng = np.random.RandomState(args.seed)
    # Sorted, not shuffled: the sample is a bag of images, so order is
    # irrelevant, but reading it in index order keeps a parquet-backed dataset
    # inside one row group at a time instead of thrashing its decode cache.
    idx = sorted(rng.permutation(len(ds))[:n].tolist())
    ds = Subset(ds, idx)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)

    labels = []
    written = 0
    for batch in tqdm(loader, desc="writing reference images"):
        x1, y = batch[1], batch[2]
        if args.eval_image_size != x1.shape[-1]:
            x1 = F.interpolate(x1, size=(args.eval_image_size, args.eval_image_size),
                               mode="area" if args.eval_image_size < x1.shape[-1]
                               else "bicubic",
                               align_corners=None if args.eval_image_size < x1.shape[-1] else False)
        for i in range(x1.shape[0]):
            Image.fromarray(to_uint8(x1[i])).save(
                os.path.join(args.out_dir, f"real_{written:07d}.png"))
            labels.append(int(y[i]))
            written += 1

    # IMPORTANT: keep --out_dir free of anything that is not an image.
    # clean-fid globs `**/*.{ext}` recursively with 'npy' among its extensions,
    # so a labels.npy sitting next to (or under) the images is loaded as an
    # image and crashes feature extraction. Side-car metadata therefore lives in
    # a *sibling* directory, which the recursive glob cannot reach.
    meta_dir = args.out_dir.rstrip("/\\") + "_meta"
    os.makedirs(meta_dir, exist_ok=True)
    np.save(os.path.join(meta_dir, "labels.npy"),
            np.asarray(labels, dtype=np.int64))
    with open(os.path.join(meta_dir, "reference_meta.json"), "w") as f:
        json.dump({
            "dataset": args.dataset,
            "split": args.split,
            "num_images": written,
            "source_image_size": args.source_image_size,
            "eval_image_size": args.eval_image_size,
            "seed": args.seed,
            "cleanfid_name": args.cleanfid_name,
            "cleanfid_mode": args.cleanfid_mode,
        }, f, indent=2)
    print(f"[done] {written} reference images -> {args.out_dir}")
    print(f"[done] labels / metadata      -> {meta_dir}")

    if args.cleanfid_name:
        from cleanfid import fid
        from evaluation.metrics import _dedup_cleanfid_extensions

        device = args.device
        if "cuda" in str(device) and not torch.cuda.is_available():
            print("[cleanfid] CUDA requested but unavailable -> using CPU")
            device = "cpu"
        # clean-fid refuses to overwrite an existing stats file -- it raises
        # "The statistics file <name> already exists", which would make every
        # re-run of this stage fail after it had already rewritten the images.
        # The stats have to describe the images just written, so the default is
        # to replace them; --reuse_stats keeps what is there.
        if cleanfid_stats_exist(args.cleanfid_name, args.cleanfid_mode):
            if args.reuse_stats:
                print(f"[cleanfid] stats '{args.cleanfid_name}' already exist "
                      f"-> reusing them (--reuse_stats)")
                print(f"[cleanfid] train with --cleanfid_dataset_name "
                      f"{args.cleanfid_name}")
                return
            print(f"[cleanfid] stats '{args.cleanfid_name}' already exist "
                  f"-> rebuilding them for the {written:,} images just written")
            try:
                fid.remove_custom_stats(args.cleanfid_name,
                                        mode=args.cleanfid_mode)
            except Exception as exc:      # already gone, or a layout we cannot guess
                print(f"[cleanfid] could not remove the old stats ({exc}); "
                      f"trying to build over them")
        print(f"[cleanfid] building custom stats '{args.cleanfid_name}' "
              f"(mode={args.cleanfid_mode}, device={device})")
        # Same Windows dedup guard the FID path uses: clean-fid globs once per
        # extension, so png/PNG match every file twice on a case-insensitive
        # filesystem and the stats would be built from a doubled file list.
        with _dedup_cleanfid_extensions(args.out_dir):
            fid.make_custom_stats(args.cleanfid_name, args.out_dir,
                                  mode=args.cleanfid_mode,
                                  batch_size=args.stats_batch_size,
                                  device=torch.device(device))
        print(f"[cleanfid] train with --cleanfid_dataset_name {args.cleanfid_name}")


if __name__ == "__main__":
    main()

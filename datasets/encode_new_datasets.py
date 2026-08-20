#!/usr/bin/env python3
"""
Pre-encode CIFAR-10 / ImageNet / ImageNet-LT into Stable-Diffusion VAE latents
in the sharded layout read by ``datasets.latent_shards.ShardedLatentDataset``.

This mirrors what ``datasets/StableVAE_EncodedDatasetCreator.py`` does for AFHQ
and what ``datasets/preprocess_imnet_dataset.py`` does for ImageNet, but writes
one consistent, label-carrying, chunked format for all three new datasets.

Examples
--------
CIFAR-10 (downloads itself):

    python datasets/encode_new_datasets.py \
        --dataset cifar10 --data_root ./data/cifar10 \
        --out_root ./data/encoded/cifar10_256 \
        --image_size 256 --batch_size 64 --device cuda:0

ImageNet-1k train split:

    python datasets/encode_new_datasets.py \
        --dataset imagenet --data_root /data/imagenet \
        --imagenet_subdir train \
        --out_root /data/encoded/imagenet_256 \
        --image_size 256 --batch_size 128 --num_workers 8 --device cuda:0

ImageNet-LT (uses the official split file when it can be fetched):

    python datasets/encode_new_datasets.py \
        --dataset imagenet-lt --data_root /data/imagenet \
        --out_root /data/encoded/imagenet_lt_256 \
        --image_size 256 --batch_size 128 --num_workers 8 --device cuda:0
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.gfm_image_datasets import build_image_dataset
from datasets.latent_shards import LatentShardWriter
from networks.networks import create_hf_vae_wrappers


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   choices=["cifar10", "imagenet", "imagenet-lt"])
    p.add_argument("--data_root", required=True,
                   help="CIFAR-10 download dir, or the ImageNet root that "
                        "contains train/ and val/")
    p.add_argument("--out_root", required=True,
                   help="Directory to write chunk_*.pt / metadata.json into")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--imagenet_subdir", default="train",
                   help="Sub-directory of --data_root holding the wnid folders")
    p.add_argument("--imagenet_lt_split_file", default=None,
                   help="Explicit path to ImageNet_LT_train.txt")
    p.add_argument("--no_pareto_fallback", action="store_true",
                   help="Fail instead of reconstructing an LT split")
    p.add_argument("--pareto_seed", type=int, default=0)

    p.add_argument("--image_size", type=int, default=256,
                   help="Resolution fed to the VAE; latents are image_size/8")
    p.add_argument("--hf_vae_model", type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--flip_augment", action="store_true",
                   help="Also encode the horizontally flipped copy of every "
                        "image (doubles the dataset, as done for AFHQ)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--chunk_size", type=int, default=8192)
    p.add_argument("--storage_dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=0,
                   help="Encode at most this many images (0 = all). Useful for "
                        "smoke tests on a laptop.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    storage_dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.storage_dtype]

    ds = build_image_dataset(
        args.dataset, args.data_root, split=args.split,
        image_size=args.image_size, use_horizontal_flips=False,
        return_label=True,
        imagenet_subdir=args.imagenet_subdir,
        split_file=args.imagenet_lt_split_file,
        allow_pareto_fallback=not args.no_pareto_fallback,
        pareto_seed=args.pareto_seed,
    )
    print(f"[data] {args.dataset}/{args.split}: {len(ds)} images "
          f"at {args.image_size}x{args.image_size}")
    if hasattr(ds, "imbalance_report"):
        print(f"[data] imbalance: {json.dumps(ds.imbalance_report(), indent=2)}")

    if args.limit and args.limit < len(ds):
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(args.limit)))
        print(f"[data] limited to {len(ds)} images")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=False)

    print(f"[vae] loading {args.hf_vae_model}")
    encoder, _ = create_hf_vae_wrappers(
        pretrained_model_name_or_path=args.hf_vae_model, device=device)
    encoder.eval()

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "image_size": args.image_size,
        "vae": args.hf_vae_model,
        "flip_augment": bool(args.flip_augment),
        "source_root": os.path.abspath(args.data_root),
    }
    if hasattr(ds, "imbalance_report"):
        meta["imbalance_report"] = ds.imbalance_report()

    writer = LatentShardWriter(args.out_root, chunk_size=args.chunk_size,
                               storage_dtype=storage_dtype, metadata=meta)

    with torch.no_grad():
        for batch in tqdm(loader, desc="encoding"):
            x1, y = batch[1], batch[2]
            x1 = x1.to(device, non_blocking=True)
            latents, _ = encoder(x1)
            writer.add(latents, y)
            if args.flip_augment:
                latents_f, _ = encoder(torch.flip(x1, dims=[3]))
                writer.add(latents_f, y)

    out_meta = writer.close()
    print(f"[done] wrote {out_meta['total_samples']} latents of shape "
          f"{out_meta['latent_shape']} into {out_meta['num_chunks']} chunks "
          f"at {args.out_root}")
    print(f"[done] train with:  --dataset {args.dataset} --use_pre_encoded "
          f"--encoded_dataset_path {args.out_root} "
          f"--latent_size {args.image_size // 8}")


if __name__ == "__main__":
    main()

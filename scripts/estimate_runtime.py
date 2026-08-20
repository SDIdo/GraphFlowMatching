#!/usr/bin/env python3
"""
Turn a short pilot run into a real wall-clock estimate.

Every run writes per-step timings to ``train_log_steps.csv`` (see
``utils/loss_logger.py``), so a few hundred steps on the target GPU is enough to
project the full job and pick a ``--time`` for Slurm. That beats any a-priori
FLOP estimate, because it captures the fp32/no-AMP path, the deterministic cuDNN
setting, dataloader stalls and the quadratic graph term as they actually run.

Usage
-----
    # after ~200 steps of a pilot run
    python scripts/estimate_runtime.py --run work/runs/cifar10/models \
        --dataset cifar10 --epochs 200 --batch_size 64

    # compare against another dataset without re-running
    python scripts/estimate_runtime.py --run work/runs/cifar10/models \
        --dataset imagenet --epochs 200 --batch_size 64
"""

import argparse
import csv
import os

# Training-set sizes actually used by this pipeline.
DATASET_SIZES = {
    "cifar10": 50_000,          # torchvision CIFAR-10 train split
    "imagenet": 1_281_167,      # ILSVRC-2012 train split
    "imagenet-lt": 115_846,     # official ImageNet_LT_train.txt (verified)
}


def read_steps(run_dir):
    path = os.path.join(run_dir, "train_log_steps.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found. Run a pilot first (--csv_log is on by default).")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(float(r["step"])), float(r["wall_time_s"])))
            except (KeyError, TypeError, ValueError):
                continue
    if len(rows) < 2:
        raise SystemExit(f"{path} has fewer than 2 usable rows; run longer.")
    return rows


def fmt(hours):
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Directory with train_log_steps.csv")
    p.add_argument("--dataset", required=True, choices=list(DATASET_SIZES))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--warmup_steps", type=int, default=20,
                   help="Ignore the first N steps (cuDNN autotune, allocator "
                        "warmup, first-epoch page cache misses)")
    p.add_argument("--slurm_time_limit_h", type=float, default=24.0,
                   help="Wall clock of one job, for the 'jobs needed' figure")
    args = p.parse_args(argv)

    rows = read_steps(args.run)
    rows = [r for r in rows if r[0] > args.warmup_steps] or rows
    (s0, t0), (s1, t1) = rows[0], rows[-1]
    if s1 <= s0:
        raise SystemExit("step numbers are not increasing; nothing to project.")
    sec_per_step = (t1 - t0) / (s1 - s0)

    n = DATASET_SIZES[args.dataset]
    steps_per_epoch = max(1, n // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    hours = total_steps * sec_per_step / 3600.0

    print(f"measured over steps {s0}..{s1} of {args.run}")
    print(f"  {sec_per_step:.3f} s/step   "
          f"({args.batch_size / sec_per_step:.1f} img/s)")
    print()
    print(f"projection for {args.dataset} ({n:,} images, batch {args.batch_size})")
    print(f"  steps/epoch        : {steps_per_epoch:,}")
    print(f"  total steps        : {total_steps:,}  ({args.epochs} epochs)")
    print(f"  estimated training : {fmt(hours)}")
    jobs = -(-hours // args.slurm_time_limit_h)  # ceil
    print(f"  jobs at --time={args.slurm_time_limit_h:g}h : {jobs:.0f}"
          + ("" if jobs <= 1 else "  (needs --retrain_flow_network to chain)"))
    print()
    print("  epochs that fit in one job:"
          f" {int(args.slurm_time_limit_h * 3600 / (steps_per_epoch * sec_per_step))}")


if __name__ == "__main__":
    main()

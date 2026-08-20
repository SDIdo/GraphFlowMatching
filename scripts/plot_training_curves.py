#!/usr/bin/env python3
"""
Plot the flow-matching training curves recorded by ``utils.loss_logger``.

Produces, for one or more runs:
  * ``loss_curve.pdf/.png``      -- per-step loss (EMA-smoothed) + per-epoch mean
  * ``loss_terms.pdf/.png``      -- diffusion vs reaction term magnitudes
  * ``fid_curve.pdf/.png``       -- in-training FID probe, when present

Example
-------
    python scripts/plot_training_curves.py \
        --run cifar10=runs/cifar10/models \
        --run imagenet-lt=runs/imagenet_lt/models \
        --out_dir paper/figures
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#2E5FA3", "#D1495B", "#3F8F6B", "#B07A2E", "#6C4F9E", "#4B4B4B"]


def read_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            try:
                cols[k].append(float(v) if v not in ("", None) else np.nan)
            except (TypeError, ValueError):
                cols[k].append(np.nan)
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def ema(x, alpha=0.02):
    if len(x) == 0:
        return x
    out = np.empty_like(x, dtype=np.float64)
    acc = x[0]
    for i, v in enumerate(x):
        if np.isfinite(v):
            acc = (1 - alpha) * acc + alpha * v
        out[i] = acc
    return out


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {os.path.join(out_dir, name)}.pdf/.png")


def auto_alpha(n, target_window=60):
    """EMA factor giving roughly a target_window-step effective window."""
    if n <= 1:
        return 1.0
    return float(min(0.5, max(0.002, target_window / float(n))))


def plot_loss(runs, out_dir, name="loss_curve", smooth=0.0, logy=True):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for i, (label, d) in enumerate(runs.items()):
        c = PALETTE[i % len(PALETTE)]
        st = d.get("steps", {})
        if "step" in st and "loss" in st:
            axes[0].plot(st["step"], st["loss"], color=c, alpha=0.18, linewidth=0.7)
            alpha = smooth if smooth > 0 else auto_alpha(len(st["loss"]))
            axes[0].plot(st["step"], ema(st["loss"], alpha), color=c,
                         linewidth=1.6, label=label)
        ep = d.get("epochs", {})
        if "epoch" in ep and "avg_loss" in ep:
            axes[1].plot(ep["epoch"], ep["avg_loss"], color=c, linewidth=1.6,
                         marker="o", markersize=2.5, label=label)
    _style(axes[0], "training step", "normalised velocity MSE",
           "Per-step training loss")
    _style(axes[1], "epoch", "mean normalised velocity MSE",
           "Per-epoch training loss")
    if logy:
        for ax in axes:
            ax.set_yscale("log")
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, name)


def plot_terms(runs, out_dir, name="loss_terms"):
    have = [(l, d) for l, d in runs.items()
            if "epoch" in d.get("epochs", {})
            and "avg_diffusion_term" in d.get("epochs", {})]
    if not have:
        print("[fig] no term data; skipping loss_terms")
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i, (label, d) in enumerate(have):
        c = PALETTE[i % len(PALETTE)]
        ep = d["epochs"]
        ax.plot(ep["epoch"], ep["avg_reaction_term"], color=c, linewidth=1.6,
                label=f"{label} -- reaction")
        ax.plot(ep["epoch"], ep["avg_diffusion_term"], color=c, linewidth=1.6,
                linestyle="--", label=f"{label} -- graph correction")
    _style(ax, "epoch", r"mean $\max|\cdot|$ per batch",
           "Reaction vs. graph-correction magnitude")
    ax.set_yscale("log")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, name)


def plot_fid(runs, out_dir, name="fid_curve"):
    have = [(l, d) for l, d in runs.items() if "step" in d.get("fid", {})]
    if not have:
        print("[fig] no FID data; skipping fid_curve")
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i, (label, d) in enumerate(have):
        c = PALETTE[i % len(PALETTE)]
        ax.plot(d["fid"]["step"], d["fid"]["fid"], color=c, linewidth=1.6,
                marker="o", markersize=2.5, label=label)
    _style(ax, "training step", "FID (in-training probe)",
           "FID during training")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, name)


def load_run(run_dir):
    return {
        "steps": read_csv(os.path.join(run_dir, "train_log_steps.csv")),
        "epochs": read_csv(os.path.join(run_dir, "train_log_epochs.csv")),
        "fid": read_csv(os.path.join(run_dir, "train_log_fid.csv")),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   metavar="LABEL=DIR",
                   help="Run directory containing train_log_*.csv, optionally "
                        "prefixed with a display label")
    p.add_argument("--out_dir", default="paper/figures")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="EMA factor for the per-step curve (0 = auto, sized to "
                        "give a ~60-step effective window)")
    p.add_argument("--no_logy", action="store_true")
    args = p.parse_args(argv)

    runs = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        if not path:
            label, path = os.path.basename(os.path.normpath(label)), label
        runs[label] = load_run(path)
        n = len(runs[label]["steps"].get("step", []))
        print(f"[data] {label}: {n} logged steps from {path}")

    plot_loss(runs, args.out_dir, smooth=args.smooth, logy=not args.no_logy)
    plot_terms(runs, args.out_dir)
    plot_fid(runs, args.out_dir)


if __name__ == "__main__":
    main()

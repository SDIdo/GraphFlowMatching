#!/usr/bin/env python3
"""
One-command driver for the CIFAR-10 / ImageNet / ImageNet-LT pipeline:
encode -> reference set -> train -> evaluate -> plot -> LaTeX tables.

Each stage is a subprocess call to the corresponding script, so any stage can
also be run by hand; ``--stages`` selects which ones to run and ``--dry_run``
prints the commands without executing them.

Examples
--------
    # everything, CIFAR-10, small budget
    python scripts/run_pipeline.py --dataset cifar10 \
        --data_root ./data/cifar10 --work_dir ./work --device cuda:0 \
        --flow_epochs 100 --train_batch_size 64 --num_eval_samples 10000

    # just re-evaluate an existing checkpoint
    python scripts/run_pipeline.py --dataset imagenet-lt \
        --data_root /data/imagenet --work_dir /data/work \
        --stages evaluate,plot,paper
"""

import argparse
import os
import shlex
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ALL_STAGES = ["encode", "reference", "train", "evaluate", "plot", "paper"]

DEFAULTS = {
    # dataset          image_size  eval_size  cleanfid stats name
    "cifar10":     (256, 32,  "cifar10_gfm_32",     "CIFAR-10"),
    "imagenet":    (256, 256, "imagenet_gfm_256",   "ImageNet"),
    "imagenet-lt": (256, 256, "imagenet_lt_gfm_256", "ImageNet-LT"),
}


def run(cmd, dry_run=False, env=None):
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n\033[1m$ {printable}\033[0m", flush=True)
    if dry_run:
        return 0
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"stage failed (exit {r.returncode}): {printable}")
    return r.returncode


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DEFAULTS))
    p.add_argument("--data_root", required=True,
                   help="CIFAR-10 download dir, or the ImageNet root")
    p.add_argument("--work_dir", default="./work",
                   help="Root for encoded latents, reference sets, runs, results")
    p.add_argument("--stages", default=",".join(ALL_STAGES),
                   help=f"Comma-separated subset of {ALL_STAGES}")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--eval_image_size", type=int, default=None)
    p.add_argument("--num_reference_images", type=int, default=50000)
    p.add_argument("--num_eval_samples", type=int, default=10000)
    p.add_argument("--encode_limit", type=int, default=0)
    p.add_argument("--imagenet_lt_split_file", default=None,
                   help="Local ImageNet_LT_train.txt. Strongly preferred on a "
                        "cluster: without it the loader tries to download the "
                        "split, and on a compute node with no outbound network "
                        "that silently falls back to a Pareto reconstruction, "
                        "which is NOT the official split.")
    p.add_argument("--no_pareto_fallback", action="store_true",
                   help="Fail instead of reconstructing an ImageNet-LT split. "
                        "Use this to guarantee the official split was used.")

    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--flow_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--base_model", default="dit",
                   choices=["dit", "pnpUNet", "adm", "resnet"])
    p.add_argument("--adj_mode", default="attention",
                   choices=["attention", "cosine", "gaussian", "knn"])
    p.add_argument("--no_diffusion", action="store_true",
                   help="Ablate the graph correction term")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nsteps", type=int, default=3)
    p.add_argument("--int_method", default="rk4")
    p.add_argument("--in_training_fid", action="store_true",
                   help="Enable the in-training FID probe (needs the reference "
                        "stage to have run first)")
    p.add_argument("--extra_train_args", default="",
                   help="Extra arguments appended verbatim to train.py")
    p.add_argument("--tag", default="",
                   help="Variant name appended to the run/result directories "
                        "(and to the paper's row label) while leaving the "
                        "encoded latents and reference set shared. Use this for "
                        "ablation sweeps so the expensive encode/reference "
                        "stages are done once.")
    return p.parse_args(argv)


def lt_args(a):
    """ImageNet-LT split options shared by the encode and reference stages."""
    out = []
    if a.dataset == "imagenet-lt":
        if a.imagenet_lt_split_file:
            out += ["--imagenet_lt_split_file", a.imagenet_lt_split_file]
        if a.no_pareto_fallback:
            out += ["--no_pareto_fallback"]
    return out


def main(argv=None):
    a = parse_args(argv)
    img, ev, stats_name, label = DEFAULTS[a.dataset]
    image_size = a.image_size or img
    eval_size = a.eval_image_size or ev
    slug = a.dataset.replace("-", "_")

    # The tag distinguishes ablation variants. It deliberately does NOT enter
    # enc_dir / ref_dir: those depend only on the dataset and resolution, so
    # every variant of a sweep reuses one encode pass and one reference set.
    tag = a.tag.strip().replace(" ", "_")
    variant = f"{slug}__{tag}" if tag else slug
    label = f"{label} ({tag})" if tag else label

    work = os.path.abspath(a.work_dir)
    enc_dir = os.path.join(work, "encoded", f"{slug}_{image_size}")
    ref_dir = os.path.join(work, "reference", f"{slug}_{eval_size}")
    run_dir = os.path.join(work, "runs", variant, "models")
    img_dir = os.path.join(work, "runs", variant, "images")
    res_dir = os.path.join(work, "results", variant)
    tmp_fid = os.path.join(work, "runs", variant, "fid_tmp")

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    for s in stages:
        if s not in ALL_STAGES:
            raise SystemExit(f"unknown stage {s!r}; choose from {ALL_STAGES}")
    py = sys.executable

    print(f"[pipeline] dataset={a.dataset} image_size={image_size} "
          f"latent={image_size // 8} eval_size={eval_size}"
          + (f" tag={tag}" if tag else ""))
    print(f"[pipeline] work_dir={work}")
    print(f"[pipeline] stages={stages}")

    if "encode" in stages:
        cmd = [py, "datasets/encode_new_datasets.py",
               "--dataset", a.dataset, "--data_root", a.data_root,
               "--out_root", enc_dir, "--image_size", str(image_size),
               "--batch_size", "64", "--num_workers", str(a.num_workers),
               "--device", a.device]
        if a.encode_limit:
            cmd += ["--limit", str(a.encode_limit)]
        cmd += lt_args(a)
        run(cmd, a.dry_run)

    if "reference" in stages:
        run([py, "datasets/make_reference_set.py",
             "--dataset", a.dataset, "--data_root", a.data_root,
             "--out_dir", ref_dir,
             "--source_image_size", str(image_size),
             "--eval_image_size", str(eval_size),
             "--num_images", str(a.num_reference_images),
             "--num_workers", str(a.num_workers),
             "--device", a.device,
             "--cleanfid_name", stats_name] + lt_args(a), a.dry_run)

    if "train" in stages:
        cmd = [py, "train.py",
               "--dataset", a.dataset,
               "--use_pre_encoded",
               "--encoded_dataset_path", enc_dir,
               "--image_size", str(image_size),
               "--model_savepath", run_dir,
               "--image_savepath", img_dir,
               "--temp_fid_comp_img_directory", tmp_fid,
               "--base_model", a.base_model,
               "--adj_mode", a.adj_mode,
               "--flow_model_type", "nonLinearHeatDiffusion2",
               "--train_batch_size", str(a.train_batch_size),
               "--flow_epochs", str(a.flow_epochs),
               "--lr", str(a.lr),
               "--num_workers", str(a.num_workers),
               "--seed", str(a.seed),
               "--nsteps", str(a.nsteps),
               "--int_method", a.int_method,
               "--device", a.device,
               "--cleanfid_dataset_name", stats_name if a.in_training_fid else "none"]
        cmd += ["--no-diffusion"] if a.no_diffusion else ["--diffusion"]
        if a.extra_train_args:
            cmd += shlex.split(a.extra_train_args)
        run(cmd, a.dry_run)

    if "evaluate" in stages:
        ckpt = os.path.join(run_dir, "vel_net_best_fid.pt")
        if not os.path.exists(ckpt):
            fallback = os.path.join(run_dir, "vel_net.pt")
            if os.path.exists(fallback) or a.dry_run:
                print(f"[pipeline] {ckpt} missing -> using {fallback}")
                ckpt = fallback
            else:
                raise SystemExit(f"no checkpoint found in {run_dir}")
        run([py, "evaluation/evaluate_gfm.py",
             "--checkpoint", ckpt,
             "--model_savepath", run_dir,
             "--dataset", a.dataset,
             "--run_name", variant,
             "--reference_dir", ref_dir,
             "--out_dir", res_dir,
             "--num_samples", str(a.num_eval_samples),
             "--batch_size", str(a.train_batch_size),
             "--latent_size", str(image_size // 8),
             "--eval_image_size", str(eval_size),
             "--base_model", a.base_model,
             "--int_method", a.int_method,
             "--nsteps", str(a.nsteps),
             "--num_workers", str(a.num_workers),
             "--device", a.device,
             "--cleanfid_dataset_name", stats_name], a.dry_run)

    if "plot" in stages:
        run([py, "scripts/plot_training_curves.py",
             "--run", f"{label}={run_dir}",
             "--out_dir", "paper/figures"], a.dry_run)

    if "paper" in stages:
        run([py, "scripts/build_paper.py",
             "--result", f"{label}={res_dir}",
             "--run", f"{label}={run_dir}"], a.dry_run)

    print("\n[pipeline] done. Build the PDF with:  cd paper && latexmk -pdf main.tex")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
End-to-end evaluation of a trained Graph Flow Matching checkpoint:
sample images, then compute FID, Inception Score, MMD (and KID) and produce the
t-SNE figure and the sample grid used by the LaTeX report.

The velocity network and the ODE integrator are used exactly as trained --
``utils.utils.integrate_ode`` is called unmodified; this script only wraps
sampling and measurement around it.

Example
-------
    python evaluation/evaluate_gfm.py \
        --checkpoint runs/cifar10/models/vel_net_best_fid.pt \
        --model_savepath runs/cifar10/models \
        --reference_dir data/reference/cifar10_32 \
        --out_dir results/cifar10 \
        --dataset cifar10 --num_samples 10000 \
        --latent_size 32 --eval_image_size 32 \
        --base_model dit --int_method rk4 --nsteps 3 --device cuda:0
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from evaluation import metrics as M

CLASS_NAMES = {
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog",
                "frog", "horse", "ship", "truck"],
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # what to evaluate
    p.add_argument("--checkpoint", required=True,
                   help="Path to vel_net_best_fid.pt / vel_net.pt")
    p.add_argument("--model_savepath", default=None,
                   help="Directory holding hf_decoder_wrapper.pt "
                        "(defaults to the checkpoint's directory)")
    p.add_argument("--hf_vae_model", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--run_name", default=None,
                   help="Label written into metrics.json (defaults to dataset)")

    # sampling
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--latent_channels", type=int, default=4)
    p.add_argument("--latent_size", type=int, default=32)
    p.add_argument("--base_model", default="dit",
                   choices=["adm", "resnet", "dit", "pnpUNet"])
    p.add_argument("--int_method", default="rk4",
                   choices=["dopri5", "rk4", "rk2", "euler"])
    p.add_argument("--nsteps", type=int, default=3)
    p.add_argument("--time_convention", default="reverse",
                   choices=["reverse", "standard"])
    p.add_argument("--eval_image_size", type=int, default=256,
                   help="Metrics are computed at this resolution (32 for the "
                        "standard CIFAR-10 protocol)")
    p.add_argument("--reuse_samples", action="store_true",
                   help="Skip sampling if the sample directory already has "
                        "enough images")

    # references / metrics
    p.add_argument("--reference_dir", required=True,
                   help="Folder of real images at --eval_image_size "
                        "(see datasets/make_reference_set.py)")
    p.add_argument("--cleanfid_dataset_name", default=None,
                   help="Registered clean-fid custom stats name; if omitted, "
                        "FID is computed folder-vs-folder")
    p.add_argument("--cleanfid_mode", default="clean")
    p.add_argument("--reference_limit", type=int, default=0,
                   help="Cap the number of reference images used for "
                        "IS/MMD/t-SNE features (0 = all)")
    p.add_argument("--is_splits", type=int, default=10)
    p.add_argument("--mmd_subset_size", type=int, default=2000)
    p.add_argument("--mmd_n_subsets", type=int, default=20)
    p.add_argument("--kid_subset_size", type=int, default=1000)
    p.add_argument("--kid_n_subsets", type=int, default=100)
    p.add_argument("--tsne_samples", type=int, default=2000)
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--skip_fid", action="store_true")
    p.add_argument("--skip_mmd", action="store_true")
    p.add_argument("--skip_tsne", action="store_true")

    # plumbing
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sample_dir", default=None,
                   help="Where generated PNGs go (default <out_dir>/samples)")
    p.add_argument("--feature_batch_size", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid_rows", type=int, default=8)
    p.add_argument("--grid_cols", type=int, default=8)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
#  Sampling
# --------------------------------------------------------------------------- #
def _fmt_hms(seconds):
    """Seconds as h:mm:ss -- these timings span minutes to hours."""
    seconds = int(round(float(seconds)))
    return f"{seconds // 3600:d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def load_decoder(args):
    dec_path = os.path.join(args.model_savepath, "hf_decoder_wrapper.pt")
    if os.path.exists(dec_path):
        print(f"[vae] loading decoder wrapper {dec_path}")
        return torch.load(dec_path, map_location=args.device, weights_only=False)
    print(f"[vae] {dec_path} missing -> instantiating {args.hf_vae_model}")
    from networks.networks import create_hf_vae_wrappers
    _, dec = create_hf_vae_wrappers(
        pretrained_model_name_or_path=args.hf_vae_model, device=args.device)
    return dec


@torch.no_grad()
def sample_images(args, vel_net, decoder, out_dir):
    """Integrate the learned velocity field and write PNGs at eval resolution.

    Returns (out_dir, elapsed_seconds). That elapsed time IS the inference
    cost of the run -- ODE integration, VAE decode and PNG write -- and is
    what metrics.json and the report tables report.
    """
    from utils.utils import integrate_ode

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dt = 1.0 / args.nsteps
    written = 0
    t0 = time.time()
    while written < args.num_samples:
        bs = min(args.batch_size, args.num_samples - written)
        z0 = torch.randn(bs, args.latent_channels, args.latent_size,
                         args.latent_size, device=args.device)
        z_final, _ = integrate_ode(
            vel_net, z0, dt, args.nsteps, method=args.int_method,
            base_model=args.base_model, traj=False,
            time_convention=args.time_convention)
        del z0

        x = decoder(z_final)                     # [-1, 1], [B, 3, H, W]
        del z_final
        if x.shape[-1] != args.eval_image_size:
            mode = "area" if args.eval_image_size < x.shape[-1] else "bicubic"
            x = F.interpolate(x, size=(args.eval_image_size, args.eval_image_size),
                              mode=mode,
                              **({} if mode == "area" else {"align_corners": False}))
        x = ((x.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).cpu()

        for i in range(x.shape[0]):
            arr = x[i].permute(1, 2, 0).numpy()
            Image.fromarray(arr).save(os.path.join(out_dir, f"gen_{written:07d}.png"))
            written += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        el = time.time() - t0
        print(f"[sample] {written}/{args.num_samples}  "
              f"({written / max(el, 1e-6):.1f} img/s)", flush=True)
    elapsed = time.time() - t0
    print(f"[sample] {written} images in {_fmt_hms(elapsed)}  "
          f"({1000.0 * elapsed / max(written, 1):.1f} ms/img at "
          f"{args.nsteps} NFE)")
    return out_dir, elapsed


def save_sample_grid(sample_dir, out_path, rows=8, cols=8):
    paths = M.list_images(sample_dir, limit=rows * cols)
    if not paths:
        return None
    tiles = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    h, w, _ = tiles[0].shape
    pad = max(2, h // 64)
    canvas = np.full(((h + pad) * rows + pad, (w + pad) * cols + pad, 3),
                     255, dtype=np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)
        y0 = pad + r * (h + pad)
        x0 = pad + c * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w] = tile
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    return out_path


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    args.model_savepath = args.model_savepath or os.path.dirname(
        os.path.abspath(args.checkpoint))
    args.sample_dir = args.sample_dir or os.path.join(args.out_dir, "samples")
    args.run_name = args.run_name or args.dataset
    os.makedirs(args.out_dir, exist_ok=True)

    if "cuda" in args.device and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable -> falling back to CPU")
        args.device = "cpu"

    t_eval0 = time.time()
    # None, not 0.0, when the samples are reused: that cost was paid by an
    # earlier job, and a zero here would read as 'sampling was free'.
    sampling_s = None
    have = len(M.list_images(args.sample_dir)) if os.path.isdir(args.sample_dir) else 0
    if args.reuse_samples and have >= args.num_samples:
        print(f"[sample] reusing {have} existing images in {args.sample_dir}")
    else:
        print(f"[model] loading {args.checkpoint}")
        from networks.networks import DecoderWrapper, EncoderWrapper
        torch.serialization.add_safe_globals([DecoderWrapper, EncoderWrapper])
        vel_net = torch.load(args.checkpoint, map_location=args.device,
                             weights_only=False)
        vel_net.eval()
        n_params = sum(p.numel() for p in vel_net.parameters())
        print(f"[model] {n_params/1e6:.2f}M parameters")
        decoder = load_decoder(args)
        decoder.eval()
        _, sampling_s = sample_images(args, vel_net, decoder, args.sample_dir)
        del vel_net, decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grid_path = save_sample_grid(
        args.sample_dir, os.path.join(args.out_dir, f"samples_{args.run_name}.png"),
        rows=args.grid_rows, cols=args.grid_cols)

    results = {
        "run_name": args.run_name,
        "dataset": args.dataset,
        "checkpoint": os.path.abspath(args.checkpoint),
        "num_samples": args.num_samples,
        "eval_image_size": args.eval_image_size,
        "sampler": {"method": args.int_method, "nsteps": args.nsteps,
                    "time_convention": args.time_convention,
                    "latent_shape": [args.latent_channels, args.latent_size,
                                     args.latent_size]},
        "reference_dir": os.path.abspath(args.reference_dir),
        "sample_dir": os.path.abspath(args.sample_dir),
        "sample_grid": grid_path,
    }

    # metrics_s and total_s are filled in at the very end, once the metric
    # phase has run; the dict is in results already, so it is one object.
    timing = {
        "sampling_s": sampling_s,
        "samples_reused": sampling_s is None,
        "ms_per_image": (None if sampling_s is None else
                         1000.0 * sampling_s / max(args.num_samples, 1)),
        "nfe_per_image": args.nsteps,
    }
    results["timing"] = timing

    # ---------------- FID ----------------
    if not args.skip_fid:
        try:
            fid_val = M.fid_cleanfid(
                args.sample_dir,
                ref_dir=None if args.cleanfid_dataset_name else args.reference_dir,
                dataset_name=args.cleanfid_dataset_name,
                mode=args.cleanfid_mode, device=args.device,
                num_workers=args.num_workers, batch_size=args.feature_batch_size)
            results["fid"] = fid_val
            results["fid_backend"] = f"clean-fid[{args.cleanfid_mode}]"
            print(f"[FID] {fid_val:.4f}  (clean-fid, {args.cleanfid_mode})")
        except Exception as exc:  # noqa: BLE001
            print(f"[FID] clean-fid unavailable/failed ({exc}); "
                  "will fall back to torchvision-Inception FID")
            results["fid_backend"] = "torchvision-inception (fallback)"

    # ---------------- shared Inception features ----------------
    print("[features] generated images")
    feats_fake, probs_fake = M.extract_inception_features(
        args.sample_dir, device=args.device, batch_size=args.feature_batch_size,
        num_workers=args.num_workers)
    print("[features] reference images")
    feats_real, probs_real = M.extract_inception_features(
        args.reference_dir, device=args.device, batch_size=args.feature_batch_size,
        num_workers=args.num_workers, limit=args.reference_limit)
    results["num_reference_images"] = int(feats_real.shape[0])

    if "fid" not in results and not args.skip_fid:
        results["fid"] = M.fid_from_features(feats_real, feats_fake)
        print(f"[FID] {results['fid']:.4f}  (torchvision-Inception fallback)")

    # ---------------- Inception Score ----------------
    is_mean, is_std = M.inception_score(probs_fake, splits=args.is_splits)
    results["inception_score"] = is_mean
    results["inception_score_std"] = is_std
    ris_mean, ris_std = M.inception_score(probs_real, splits=args.is_splits)
    results["inception_score_real"] = ris_mean
    results["inception_score_real_std"] = ris_std
    print(f"[IS]  {is_mean:.4f} +/- {is_std:.4f}   (real reference: "
          f"{ris_mean:.4f} +/- {ris_std:.4f})")

    # ---------------- MMD / KID ----------------
    if not args.skip_mmd:
        mmd = M.compute_mmd(feats_real, feats_fake, device=args.device,
                            subset_size=args.mmd_subset_size,
                            n_subsets=args.mmd_n_subsets, seed=args.seed)
        results["mmd"] = mmd
        print(f"[MMD] {mmd['mmd']:.5f}  (MMD^2 = {mmd['mmd2']:.3e} "
              f"+/- {mmd['mmd2_std']:.1e})")
        try:
            kid = M.compute_kid(feats_real, feats_fake,
                                subset_size=args.kid_subset_size,
                                n_subsets=args.kid_n_subsets,
                                device=args.device, seed=args.seed)
            results["kid"] = kid
            print(f"[KID] {kid['kid']:.5f} +/- {kid['kid_std']:.5f}")
        except Exception as exc:  # noqa: BLE001
            print(f"[KID] skipped ({exc})")

    # ---------------- t-SNE ----------------
    if not args.skip_tsne:
        try:
            emb, is_real, ir, _ = M.tsne_embedding(
                feats_real, feats_fake, n_samples=args.tsne_samples,
                perplexity=args.tsne_perplexity, seed=args.seed)
            fig_path = os.path.join(args.out_dir, f"tsne_{args.run_name}.png")
            M.plot_tsne(emb, is_real, fig_path,
                        title=f"t-SNE of Inception features -- {args.dataset}")
            results["tsne_figure"] = fig_path
            np.savez_compressed(
                os.path.join(args.out_dir, f"tsne_{args.run_name}.npz"),
                emb=emb, is_real=is_real)
            print(f"[t-SNE] {fig_path}")

            # Labels live in the sibling <reference_dir>_meta directory so that
            # clean-fid's recursive glob never sees a .npy inside the image
            # folder; older layouts kept them alongside the images.
            lab_path = os.path.join(args.reference_dir.rstrip("/\\") + "_meta",
                                    "labels.npy")
            if not os.path.exists(lab_path):
                lab_path = os.path.join(args.reference_dir, "labels.npy")
            names = CLASS_NAMES.get(args.dataset)
            if names and os.path.exists(lab_path):
                all_lab = np.load(lab_path)
                if args.reference_limit:
                    all_lab = all_lab[:args.reference_limit]
                if len(all_lab) >= feats_real.shape[0]:
                    fig2 = os.path.join(args.out_dir,
                                        f"tsne_{args.run_name}_byclass.png")
                    M.plot_tsne_by_class(
                        emb, is_real, all_lab[ir], names, fig2,
                        title=f"t-SNE by class -- {args.dataset}")
                    results["tsne_byclass_figure"] = fig2
                    print(f"[t-SNE] {fig2}")
        except Exception as exc:  # noqa: BLE001
            print(f"[t-SNE] skipped ({exc})")

    total_s = time.time() - t_eval0
    timing["metrics_s"] = total_s - (sampling_s or 0.0)
    timing["total_s"] = total_s

    out_json = os.path.join(args.out_dir, "metrics.json")
    M.save_json(results, out_json)
    print()
    print(f"[time] sampling {args.num_samples} images: "
          f"{'reused' if sampling_s is None else _fmt_hms(sampling_s)}   "
          f"metrics: {_fmt_hms(timing['metrics_s'])}   "
          f"total: {_fmt_hms(total_s)}")
    print(f"\n[done] metrics -> {out_json}")
    print(json.dumps({k: v for k, v in results.items()
                      if k in ("fid", "inception_score", "inception_score_std")},
                     indent=2))
    return results


if __name__ == "__main__":
    main()

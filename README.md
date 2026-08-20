# GraphFlowMatching (AAAI 2026)

Official code release for the AAAI 2026 paper **Graph Flow Matching: Enhancing Image Generation with Neighbor-Aware Flow**.

Paper (arXiv): https://arxiv.org/abs/2505.24434

This repository implements **Graph Flow Matching (GFM)**: flow matching in **VAE latent space**, augmented with a **graph-based correction** term that couples each sample to its neighbors via an adjacency matrix (e.g., attention / cosine / kNN).

---

## Setup

Create the environment from the provided Conda file:

```bash
conda env create -f environment.yml
conda activate flowmatch
```

---

## Repository Structure

- `train.py` : main training script
- `datasets/StableVAE_EncodedDatasetCreator.py` : pre-encode datasets into Stable VAE latents (for faster training)
- `datasets/CleanFIDCustomStatsCreator.py` : create CleanFID custom statistics for datasets
- `datasets/` : dataset loaders and preprocessing utilities
- `networks/` : model definitions (velocity networks, hybrid models, backbones)

---

## Pre-encode Dataset (Stable VAE Latents)

To avoid encoding images during every training iteration, pre-encode a dataset into Stable VAE latents. Here's how we did it (similar for the other datasets):

```bash
python datasets/StableVAE_EncodedDatasetCreator.py
```

After encoding, train using:

- `--use_pre_encoded`
- `--encoded_dataset_path /path/to/encoded_dataset`

---

## CleanFID Custom Dataset Stats

To compute FID using CleanFID with custom stats, create the dataset statistics first:

```bash
python datasets/CleanFIDCustomStatsCreator.py
```

Then use the dataset name during training via:

- `--cleanfid_dataset_name <name>`

---

## Training

Run training with:

```bash
python train.py
```

Example (pre-encoded latents):

```bash
python train.py   --dataset lsun_bedrooms   --use_pre_encoded   --encoded_dataset_path /path/to/encoded_dataset   --base_model dit   --adj_mode attention   --flow_model_type nonLinearHeatDiffusion2   --train_batch_size 50   --device cuda:0
```

---

---

## CIFAR-10 / ImageNet / ImageNet-LT

The released configurations target four single-domain 256x256 datasets. The
files below add CIFAR-10, ImageNet-1k and ImageNet-LT **without changing the
generative model** - the velocity network, the attention adjacency generator,
the non-linear heat-diffusion correction, the flow-matching loss and the ODE
solver are all used exactly as released. Only data loading, configuration and
measurement are new.

| File | Purpose |
| --- | --- |
| `datasets/gfm_image_datasets.py` | CIFAR-10 / ImageNet / ImageNet-LT image datasets (`(x0, x1, y)` in `[-1,1]`) |
| `datasets/latent_shards.py` | Labelled sharded latent format + chunk-aware batch sampler |
| `datasets/encode_new_datasets.py` | Pre-encode any of the three into SD-VAE latents |
| `datasets/make_reference_set.py` | Reference image folder + clean-fid custom stats |
| `evaluation/metrics.py` | FID, Inception Score, MMD, KID, t-SNE |
| `evaluation/evaluate_gfm.py` | Sample a checkpoint and compute every metric |
| `utils/loss_logger.py` | CSV mirror of the training curves (no wandb needed) |
| `scripts/plot_training_curves.py` | Loss / term-magnitude / FID figures |
| `scripts/build_paper.py` | Fills `paper/generated/*.tex` from the run artefacts |
| `scripts/run_pipeline.py` | One-command driver for the whole pipeline |
| `paper/main.tex` | The LaTeX report |

### Resolution

The SD VAE downsamples by 8, so `--image_size 256` gives the `4 x 32 x 32`
latents the released configuration expects. CIFAR-10 (natively 32x32) is
bicubically upsampled to 256 before encoding and generated samples are
downsampled back to 32x32 for measurement, which keeps the model identical and
follows the standard CIFAR-10 FID protocol.

**Constraint (model-side, not ours):** `--adj_mode attention` -- the paper's
default -- can only run at a 32x32 latent grid, i.e. `--image_size 256`. Its
adjacency generator pools by a fixed factor of 4 and flattens a hard-coded 8x8
map. `train.py` checks this and exits with an explanatory message rather than
failing deep inside the attention projection. The resolution-agnostic modes
`--adj_mode cosine` and `--adj_mode knn` run at any latent size (16x16, 8x8,
... verified), so `--image_size 128` works with those. `--adj_mode gaussian` is
numerically degenerate at these latent dimensionalities (the kernel underflows
to 0 and `sqrt(A)` yields NaN gradients); `train.py` warns about it.

### Quick start

```bash
# 0) everything at once
python scripts/run_pipeline.py --dataset cifar10 \
    --data_root ./data/cifar10 --work_dir ./work --device cuda:0

# ...or stage by stage:

# 1) pre-encode into sharded latents
python datasets/encode_new_datasets.py --dataset cifar10 \
    --data_root ./data/cifar10 --out_root ./work/encoded/cifar10_256 \
    --image_size 256 --batch_size 64 --device cuda:0

# 2) reference images + clean-fid custom statistics
python datasets/make_reference_set.py --dataset cifar10 \
    --data_root ./data/cifar10 --out_dir ./work/reference/cifar10_32 \
    --source_image_size 256 --eval_image_size 32 \
    --num_images 50000 --cleanfid_name cifar10_gfm_32

# 3) train (model unchanged; only the configuration is dataset-specific)
python train.py --dataset cifar10 --use_pre_encoded \
    --encoded_dataset_path ./work/encoded/cifar10_256 \
    --image_size 256 --base_model dit --diffusion --adj_mode attention \
    --flow_model_type nonLinearHeatDiffusion2 \
    --model_savepath ./work/runs/cifar10/models \
    --image_savepath ./work/runs/cifar10/images \
    --train_batch_size 64 --flow_epochs 200 \
    --cleanfid_dataset_name none --device cuda:0

# 4) evaluate: FID + IS + MMD + KID + t-SNE + sample grid
python evaluation/evaluate_gfm.py \
    --checkpoint ./work/runs/cifar10/models/vel_net_best_fid.pt \
    --model_savepath ./work/runs/cifar10/models \
    --dataset cifar10 --run_name cifar10 \
    --reference_dir ./work/reference/cifar10_32 \
    --out_dir ./work/results/cifar10 \
    --num_samples 10000 --latent_size 32 --eval_image_size 32 \
    --cleanfid_dataset_name cifar10_gfm_32 --device cuda:0

# 5) figures + LaTeX tables + PDF
python scripts/plot_training_curves.py --run CIFAR-10=./work/runs/cifar10/models
python scripts/build_paper.py --result CIFAR-10=./work/results/cifar10 \
                              --run CIFAR-10=./work/runs/cifar10/models
cd paper && latexmk -pdf main.tex
```

For ImageNet / ImageNet-LT replace `--dataset` and point `--data_root` at the
ImageNet root that contains `train/` and `val/`. ImageNet-LT downloads the
official `ImageNet_LT_train.txt`; if no mirror is reachable it falls back to a
deterministic Pareto(alpha=6) reconstruction with the same count profile
(1280 head / 5 tail, 1000 classes) and records which one was used in the encoded
dataset's `metadata.json`.

### New training flags

- `--image_size` : resolution fed to the VAE (latent grid is `image_size/8`)
- `--latent_size` : latent grid; auto-derived, and overridden by the encoded data
- `--latent_cache_chunks` : latent shards held in RAM
- `--imagenet_subdir`, `--imagenet_lt_split_file`, `--no_pareto_fallback`, `--pareto_seed`
- `--cleanfid_dataset_name none` : disable the in-training FID probe
- `--save_every_steps` : checkpoint independently of the FID probe
- `--csv_log` / `--no-csv_log` : write `train_log_{steps,epochs,fid}.csv`
- `--wandb_project` : override the wandb project name

### GPU

Every stage runs on GPU and takes `--device` (`train.py`,
`encode_new_datasets.py`, `make_reference_set.py`, `evaluate_gfm.py`);
`run_pipeline.py --device cuda:0` forwards it to all of them. Two notes:

- `train.py` inherits the upstream default `--device cuda:1`, which fails on a
  single-GPU machine. It is now validated up front and reports the GPU name and
  memory instead of dying with "invalid device ordinal" mid-construction.
- `clean-fid` defaults its Inception pass to `cuda:0` regardless of what the
  rest of the run uses, so `make_reference_set.py` passes `--device` through
  explicitly (and falls back to CPU when CUDA is unavailable).

### Running on a Slurm cluster

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# once, somewhere with network access -- caches the SD-VAE + Inception weights
sbatch sbatch/run_gfm.sbatch cifar10 warmup

sbatch sbatch/run_gfm.sbatch cifar10                    # full pipeline
sbatch sbatch/run_gfm.sbatch imagenet-lt encode         # one stage
sbatch --time=72:00:00 sbatch/run_gfm.sbatch imagenet   # longer budget
sbatch sbatch/run_gfm.sbatch cifar10 sweep              # graph-correction ablation
```

Submit from the repo root. Point `DATA_ROOT_IMAGENET` at the ImageNet root and
`WORK_DIR` at scratch; `HF_HOME`/`TORCH_HOME` default under `WORK_DIR` so model
downloads do not hit your home quota. The job requests one GPU constrained to
`rtx_3090|rtx_4090|rtx_6000|rtx_pro_6000` and always passes `--device cuda:0`,
since Slurm remaps the allocated card to index 0.

The `sweep` phase runs encode/reference once, then trains and evaluates four
variants (attention / cosine / kNN adjacency, plus a no-graph baseline) via
`run_pipeline.py --tag`, and builds one comparison table from all of them.

### Notes / gotchas

- **Keep the reference folder pure images.** `clean-fid` globs `**/*.{ext}`
  recursively and counts `npy` as an image extension, so a stray `labels.npy`
  anywhere under the reference tree crashes feature extraction.
  `make_reference_set.py` therefore writes labels and metadata to a *sibling*
  `<out_dir>_meta/` directory, and `evaluation/metrics.py` raises an explicit
  error if it ever finds `.npy` under an image folder.
- **Windows double-counting.** `clean-fid` globs once per entry of its
  extension set, which contains both `png` and `PNG`; on a case-insensitive
  filesystem every file matches twice. The FID wrapper narrows the extension set
  to the suffixes actually present for the duration of the call, so counts and
  compute are correct.
- **`--num_workers > 0` on Windows** uses spawn, so anything passed to a
  DataLoader must be picklable; the metrics code uses `functools.partial`
  rather than a lambda for its collate function.
- **In-training FID is off by default for these datasets** in the example
  commands (`--cleanfid_dataset_name none`), because it needs the reference
  stage to have run first. Checkpoints are still written every
  `--save_every_steps` steps.

### Extra dependencies

`clean-fid`, `scipy`, `scikit-learn`, `matplotlib` and `tqdm` on top of
`environment.yml`. `clean-fid` is optional - FID falls back to a torchvision
Inception implementation, which is reported as such and is *not* numerically
comparable to clean-fid values.


## Common Flags

- `--dataset {ffhq,lsun_bedrooms,lsun_church,celeba-hq,AFHQ-Cat-Full-256}`
- `--use_pre_encoded` and `--encoded_dataset_path`
- `--base_model {dit,adm,resnet,pnpUNet}`
- `--adj_mode {attention,cosine,gaussian,knn}` (use `--knn_k` for kNN)
- `--diffusion` / `--no-diffusion`
- `--use_wandb`
- `--use_pretrained`

---

## Outputs

The script saves:
- model checkpoints to `--model_savepath`
- generated samples (and optional reconstructions) to `--image_savepath`

---

## Citation

If you use this repository, please cite the AAAI 2026 paper:

```bibtex
@misc{siddiqui2025graphflowmatchingenhancing,
      title={Graph Flow Matching: Enhancing Image Generation with Neighbor-Aware Flow Fields}, 
      author={Md Shahriar Rahim Siddiqui and Moshe Eliasof and Eldad Haber},
      year={2025},
      eprint={2505.24434},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.24434}, 
}
```

---

## License

Research use only. See `LICENSE` if included.

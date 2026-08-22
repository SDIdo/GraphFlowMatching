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

**"torch reports no CUDA device".** Three different problems produce that line,
so the Slurm preflight now separates them and prints the evidence (node,
`CUDA_VISIBLE_DEVICES`, `SLURM_JOB_GPUS`, `/dev/nvidia*`, and what `nvidia-smi`
sees) before exiting:

- a **CPU-only wheel** (`torch.version.cuda is None`) -- reinstall from the
  cu124 index;
- **no GPU in the allocation** -- `nvidia-smi` sees nothing either; check
  `--gres`/`--partition`;
- **a GPU that CUDA cannot initialise** -- `nvidia-smi` lists the card but torch
  warns `CUDA initialization: CUDA unknown error ... Setting the available
  devices to be zero`. That is a broken node (wedged driver, missing
  `/dev/nvidia-uvm`, or a driver older than 525, which the cu124 wheels
  require), not a repo or venv problem. Resubmit elsewhere with
  `sbatch --exclude=<node> ...` and add the node to `WANT_EXCLUDE` in
  `sbatch/submit.sh`.

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

**Job names.** Every job names itself `gfm-<dataset>-<phase>` --
`gfm-inlt-erte`, `gfm-in1k-trn`, `gfm-c10-all` -- so parallel runs are
distinguishable in `squeue` instead of being five identical `gfm` rows. The
dataset abbreviates to `c10` / `in1k` / `inlt`; a single phase to three letters
(`enc`, `ref`, `trn`, `evl`, `wrm`, `rpt`, `all`, `swp`), several to their
initials (`encode,reference,train,evaluate` -> `erte`).

Slurm cannot see the positional arguments when it reads the `#SBATCH`
directives, so plain `sbatch` submissions rename themselves at startup via
`scontrol`; `sbatch/submit.sh` sets the name at submission instead, which is
better -- it is right while the job is still PENDING, and the log files are
named after it too. An explicit `-J` always wins.

`squeue` truncates NAME to 8 characters by default, so widen it:

```bash
alias sq='squeue -u $USER -o "%.10i %.20j %.2t %.11M %.6D %R"'
```

**Node exclusion.** `run_gfm.sbatch` deliberately does *not* carry an
`#SBATCH --exclude` directive: Slurm validates node names at submission time and
rejects the whole job if one is stale (`Invalid node name specified`). Use the
wrapper, which keeps only the names that actually exist:

```bash
bash sbatch/submit.sh --check                 # audit names + GPU features
bash sbatch/submit.sh --dry-run cifar10 warmup
bash sbatch/submit.sh --time=36:00:00 cifar10 # submit
```

Invoking it as `bash sbatch/submit.sh` always works. `./sbatch/submit.sh` needs
the executable bit, which is recorded in git (mode 100755) but will not survive
a copy from a Windows filesystem -- `chmod +x sbatch/submit.sh` on the cluster if
you see `Permission denied`.

Edit the list in `sbatch/submit.sh` (`WANT_EXCLUDE`), not in the job script.
Plain `sbatch sbatch/run_gfm.sbatch ...` also works -- `--constraint` already
pins the job to the four wanted GPU types, so the exclusion is belt-and-braces.

Submit from the repo root. `WORK_DIR` should point at scratch;
`HF_HOME`/`TORCH_HOME`/`GFM_CACHE_DIR` default under it so downloads do not hit
your home quota.

**ImageNet is never downloaded automatically** and lives somewhere different on
every cluster, so `DATA_ROOT_IMAGENET` has no default -- set it explicitly:

```bash
DATA_ROOT_IMAGENET=/path/to/imagenet   sbatch --time=72:00:00 sbatch/run_gfm.sbatch imagenet-lt
```

It must contain the 1000 wnid folders, directly or under `train/`. To locate an
existing copy: `find / -maxdepth 4 -type d -name n01440764 2>/dev/null | head`
(that is the first ImageNet class, so finding it locates the tree). The job
checks the path is readable before doing anything expensive. CIFAR-10 needs
none of this -- it downloads itself.

**Only `encode` and `reference` read the raw tree.** `train` works off the
encoded latents, `evaluate` off the pre-built reference set, and
`plot`/`paper`/`report` off the run directories -- so once the latents exist,

```bash
sbatch --time=96:00:00 sbatch/run_gfm.sbatch imagenet train,evaluate
```

needs no `DATA_ROOT_IMAGENET` at all, and the job no longer refuses to start
without one. `all` and `sweep` include `encode`/`reference`, so they still do.

When the root is not given, two places are consulted before giving up: the
`$WORK_DIR/.imagenet_root` memo written after a successful check, and
`source_root` in the `metadata.json` that `encode_new_datasets.py` writes beside
the latents -- so a tree that was encoded once stays findable. An explicit value
always wins; `sbatch/site.env` is the place to make it permanent.

**ImageNet as parquet shards.** If `DATA_ROOT_IMAGENET` holds HuggingFace
parquet shards (`data/train-00000-of-00294.parquet`, ...) instead of a JPEG
folder tree, that is detected automatically -- no conversion needed. If the root
holds **both**, full ImageNet uses the shards (it takes every row, so names are
irrelevant) while ImageNet-LT prefers the folder tree, whose filenames the
official split can always address. Check what you have first:

```bash
python datasets/inspect_parquet.py --root /path/to/ImageNet/data
```

The decisive question it answers is whether the shards kept the **original
filenames**. ImageNet-LT selects ~115k specific images by name, so:

- filenames present -> the official split is matched onto parquet rows exactly;
- filenames absent -> only a Pareto reconstruction from the labels is possible,
  which reproduces the long-tailed profile but not the official image list, and
  is reported as `pareto_reconstruction_parquet` in the metadata. Pass
  `--no_pareto_fallback` to refuse it instead.

Matching is on the **lowercased, extension-less basename** (`n01440764_190`),
which is unique across ImageNet, so the cosmetic rewrites a conversion applies
-- `.JPEG` -> `.jpg`, a kept or dropped `train/n01440764/` prefix, flipped path
separators -- do not break it. Rows are additionally indexed under the name with
a trailing duplicate wnid removed, since one HF conversion in circulation names
its rows `n03954731_53652_n03954731.JPEG`; no genuine ImageNet name ends in
`_n########`, so undoing that decoration cannot shadow a real file. If the match still fails, the error prints the
names it found beside the names the split asked for, so you can see the shape of
the mismatch rather than guess. Shards that renumber their rows (`0.jpg`,
`1.jpg`, ...) carry no way back to the official split: either encode from a
JPEG-folder ImageNet, or opt in to the reconstruction with
`GFM_ALLOW_PARETO=1 sbatch ...` (which drops `--no_pareto_fallback`) and accept
that the numbers are not comparable to published ImageNet-LT results.

Indexing reads only the label and filename columns, never the image bytes
(~0.5% of the data), so scanning 294 shards costs seconds. Batches are kept
inside one parquet row group so decoding stays sequential.

**ImageNet-LT split file.** `DATA_ROOT_IMAGENET` is the *image* root; the split
`.txt` files are a separate thing, listing paths like
`train/n01440764/n01440764_190.JPEG` relative to that image root. The job
auto-detects a local `ImageNet_LT_train.txt` in the usual places
(`$DATA_ROOT/ImageNet_LT/`, `$DATA_ROOT/`, `$DATA_ROOT/../ImageNet_LT/`,
`$DATA_ROOT/splits/`) and otherwise takes `IMAGENET_LT_SPLIT`:

```bash
DATA_ROOT_IMAGENET=/groups/eliasof_group/sananest/ImageNet   sbatch --time=72:00:00 sbatch/run_gfm.sbatch imagenet-lt
```

Using a local file matters: without one the loader tries to *download* the
split, and on a compute node with no outbound network it falls back to a
Pareto reconstruction -- same count profile, different file list, so the numbers
would not be comparable to published ImageNet-LT results. When a local file is
found the job also passes `--no_pareto_fallback`, so that reconstruction can
never happen silently. The loader verifies the split against the published
statistics (115,846 images / 1000 classes / 1280 head / 5 tail) and checks that
its paths actually resolve under the image root before encoding starts. The job requests one GPU constrained to
`rtx_3090|rtx_4090|rtx_6000|rtx_pro_6000` and always passes `--device cuda:0`,
since Slurm remaps the allocated card to index 0.

### GPU memory

With the graph correction on, the diffusion term attends over the whole 32x32
latent grid, so each attention block keeps a `[batch, 1024, 1024]` map alive for
the backward pass. Peak memory is therefore linear in the batch and the same for
all three datasets (at `--image_size 256` they all encode to 4x32x32 latents).
Batch 64 wants ~24 GB, which is why a 3090/4090 dies a few steps in with

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 MiB
```

So the batch is split in two. `BATCH_SIZE` (default 64) is the **effective**
batch -- what the optimizer sees, the number to keep fixed. `MICRO_BATCH`
(default `auto`) is what one forward/backward holds, i.e. what has to fit in
VRAM; `train.py` accumulates `BATCH_SIZE / MICRO_BATCH` of them per optimizer
step via `--grad_accum_steps`, so the gradient is unchanged and everything keyed
to "steps" (cosine LR schedule, `--save_every_steps`, `--checkFID_every_steps`,
the CSV/wandb curves) still counts optimizer steps.

`auto` reads the card and picks 64 on >=40 GB, 32 on 24 GB, 16 on 16 GB, 8 on
12 GB. Override either:

```bash
MICRO_BATCH=16 sbatch --time=72:00:00 sbatch/run_gfm.sbatch imagenet
BATCH_SIZE=128 MICRO_BATCH=32 sbatch sbatch/run_gfm.sbatch cifar10   # 4x accumulation
```

Running `train.py` directly, the same thing is `--train_batch_size 32
--grad_accum_steps 2`. It prints its own estimate at startup and warns before
the run if the micro-batch cannot fit, and any CUDA OOM traceback is followed by
the micro-batch/accumulation split that would have fitted. The job also exports
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to keep the smaller,
shorter-lived allocations from fragmenting the pool.

Accumulating costs perhaps 5-15% throughput (the GPU is fed in smaller pieces)
and nothing in accuracy.

### Runtime

Training is pure fp32 (no AMP in `train.py`), ~8.8 TFLOP/step for the default
DiT-B/2 reaction net at batch 64. Steps per epoch: CIFAR-10 781, ImageNet-LT
1,810, ImageNet 20,018 (optimizer steps -- accumulation does not change them).
Rough 200-epoch training times, assuming the batch fits in one piece; add
~5-15% on a card that has to accumulate:

| dataset | rtx_3090 | rtx_4090 / 6000 | rtx_pro_6000 |
| --- | --- | --- | --- |
| CIFAR-10 | ~28 h | ~13 h | ~9 h |
| ImageNet-LT | ~2.7 days | ~30 h | ~20 h |
| ImageNet | ~30 days | ~14 days | ~9 days |

One-off stages: encode ~10 min / ~30 min / 2-4 h (latents 0.4 / 0.9 / 10.5 GB),
reference ~10-30 min, evaluate ~20-35 min. **The 24 h default `--time` is not
enough for any 200-epoch run** -- pass `--time=36:00:00` for CIFAR-10,
`--time=72:00:00` for ImageNet-LT, and cut `EPOCHS` for ImageNet.

Rather than trusting the table, measure it: every run logs per-step wall times,
so after a ~200-step pilot run

```bash
python scripts/estimate_runtime.py --run work/runs/cifar10/models     --dataset cifar10 --epochs 200 --batch_size 64
```

projects the full job and tells you how many epochs fit in one allocation.

### Runs longer than one allocation

`train.py` writes `train_state.pt` next to the checkpoints -- model, AdamW
moments, cosine LR schedule, epoch and step -- every `--state_every_steps`
(default 2000) and at every epoch boundary, with an atomic `tmp` + `os.replace`
so a job killed mid-write cannot corrupt it. `--resume auto` (the default) picks
it up, so an interrupted run continues instead of restarting; resuming rewinds
to the start of the epoch that was in progress, repeating at most one epoch.

This is what `--retrain_flow_network` does *not* do: it reloads weights only, so
the optimizer moments and the LR schedule restart each time and N chained jobs
behave differently from one job of the same length.

So when the run is longer than the cluster's wall-clock cap, chain it:

```bash
# check the real cap first -- one long job always beats a chain
sinfo -o '%20P %10l %10L'                 # partition MaxTime / DefaultTime
sacctmgr show qos format=Name,MaxWall     # QOS caps
sacct -j <jobid> -X -o JobID,Timelimit,Elapsed,State

# 17 x 4 h = 68 h, enough for 200 epochs of ImageNet-LT
bash sbatch/chain.sh 17 --time=04:00:00 imagenet-lt train

# then evaluate once, at the end
sbatch --dependency=afterany:<last-id> sbatch/run_gfm.sbatch imagenet-lt evaluate

# a --dependency passed to chain.sh applies to link 1 only, so a chain can
# start after the encode job; links 2..N still wait on the link before them
bash sbatch/chain.sh 4 --time=03:55:00 --dependency=afterany:<encode-id>     imagenet-lt train
```

**The 25-epoch runs, all three datasets, under a 4 h cap.** One submission
does everything -- three datasets to 25 epochs, evaluation, and the report:

```bash
sbatch sbatch/run_all25.sbatch
```

That expands into a three-task job array, one dataset per task, running in
parallel. Each task works until Slurm kills it at 3 h 55, and **submits its own
successor before it starts**, so training simply continues for as long as it
needs; the successor resumes from `train_state.pt`. When a task finds its
dataset at 25 epochs it evaluates it, and the last task to finish builds the
report. A link with nothing left to do exits in seconds without submitting
anything, which is how the chain ends -- there is exactly one no-op job per
dataset at the end, and no link count to guess.

```bash
squeue -u $USER -o '%.10i %.20j %.2t %.11M %.20E %R'   # NAME shows the stage
tail -f sbatch/logs/gfm25_*.out
touch $WORK_DIR/.gfm25/stop                            # stop after the current links
```

It re-checks the artefacts every link, so a dataset that is already encoded, or
already trained by an earlier attempt, is not redone; resubmitting the same
command over finished work runs no stage at all. Two guards keep a broken run
from burning the queue: `GFM_MAX_STALLS` (default 3) stops a dataset whose
links make no measurable progress -- a crash loop, as opposed to slow training,
which still advances the step counter -- and `GFM_MAX_LINKS` (default 60) caps
the chain outright. A dataset that gives up still lets the report be built,
showing `--` for what it never produced. `GFM_DATASETS` and `--array` select a
subset:

```bash
GFM_DATASETS=cifar10 sbatch --array=0 sbatch/run_all25.sbatch
```

`sbatch/run25.sh` is the alternative for a cluster where a job may not submit
jobs: it pre-submits a fixed chain from the login node, and therefore has to
*guess* the link count (2 / 4 / 28 for CIFAR-10 / ImageNet-LT / ImageNet,
sized for the slowest card the constraint allows).

```bash
bash sbatch/run25.sh                 # cifar10 + imagenet-lt + imagenet
bash sbatch/run25.sh --dry-run       # print the graph, submit nothing

# fewer links (and a third of the wall-clock) by pinning the fast cards
EXTRA=--constraint=rtx_pro_6000 LINKS_IMAGENET=8 bash sbatch/run25.sh imagenet
```

Extra links there are nearly free (the epoch loop is empty and the job exits in
minutes) and a chain that runs short is not, so over-provision;
`SKIP_WARMUP=1 SKIP_ENCODE=1` resubmits just training and evaluation. Either
way, the one stage that cannot be chained is `encode`, which has no resume: it
gets 16 CPUs in `run_all25.sbatch` and its own 32-CPU job in `run25.sh`,
because JPEG decode is what makes ImageNet encode 2-4 h.

Each link resubmits the same command with `--dependency=afterany` on the
previous one -- `afterany`, not `afterok`, because a link that TIMEOUTs exits
non-zero and the next one still has to start. Chain the `train` phase only:
`encode`/`reference` are one-off, and `evaluate` belongs at the end.

Because resuming is automatic, a stale `train_state.pt` in a run directory will
be picked up by a later run too. Pass `--resume off` (or use `--tag` for a
separate directory) when you mean to start over; the run also prints a warning
if the effective batch changed across a resume.

**Running the three datasets in parallel.** They share nothing that matters --
encoded latents, reference sets, checkpoints and results are all keyed by
dataset. Submit them simultaneously, but stop before the aggregation stages:

```bash
sbatch sbatch/run_gfm.sbatch cifar10 warmup          # ONCE, let it finish first

sbatch --time=36:00:00 sbatch/run_gfm.sbatch cifar10     encode,reference,train,evaluate
sbatch --time=72:00:00 sbatch/run_gfm.sbatch imagenet-lt encode,reference,train,evaluate
EPOCHS=25 sbatch --time=96:00:00 sbatch/run_gfm.sbatch imagenet encode,reference,train,evaluate

# once they have all finished (no GPU needed, fine on a login node):
bash sbatch/run_gfm.sbatch cifar10 report
```

`report` scans `WORK_DIR`, finds every run, and rebuilds the figures and tables
from all of them at once; datasets that have not finished show as `--`. It also
prints a plain-text summary of every run -- FID, IS, epochs, training hours,
inference minutes -- so the numbers are readable straight from the job log.

**Where the timings come from.** The results table carries two wall-clock
columns beside the metrics:

| Column | Source | Covers |
| --- | --- | --- |
| `Train (h)` | `wall_time_s` in `train_log_epochs.csv` | The whole run. On a resume the logger seeds its clock from the largest value already in the CSV, so a training split over 28 chained jobs reports one cumulative figure rather than the last link's. Time spent queued between links is not counted. |
| `Infer (min)` | `timing.sampling_s` in `metrics.json` | Sampling the `N_gen` images -- ODE integration, VAE decode, PNG write. Metric computation is excluded and lives in `timing.metrics_s`; `timing.total_s` is the whole `evaluate` stage. |

`metrics.json` also records `timing.ms_per_image` and `timing.nfe_per_image`,
which is the pair to quote when comparing sampling cost across solvers or step
counts. Re-running `evaluate` with `--reuse_samples` leaves `sampling_s` null
and the table cell `--`: that run did not pay the sampling cost, and a `0` would
read as though inference were free.

Two stages are *not* safe to run concurrently, which is why the parallel
submissions above stop at `evaluate`: `plot` and `paper` are global rather than
per-dataset. `plot_training_curves.py` always writes `loss_curve` /
`loss_terms` / `fid_curve` under the same filenames, and `build_paper.py`
rewrites `results_table.tex` containing only the runs handed to it -- so two
concurrent `all` jobs would each erase the other's rows. Run `warmup` first too,
or parallel cold starts race on the same `HF_HOME` download.

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
- **The reference stage is re-runnable.** `clean-fid` refuses to overwrite an
  existing custom-stats file (`The statistics file <name> already exists. Use
  remove_custom_stats ...`), which used to make any second run of `reference`
  crash *after* it had rewritten all 50k images. The stats are now removed and
  rebuilt so they always describe the images just written; pass `--reuse_stats`
  to keep the existing ones when the images have not changed.
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

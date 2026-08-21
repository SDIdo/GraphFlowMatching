#!/bin/bash
# Placeholder only -- the script renames the job to gfm-<dataset>-<phase> once
# it can see its arguments (see the "job name" section below). Submitting via
# sbatch/submit.sh sets the final name up front, which also names the logs.
#SBATCH --job-name=gfm
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# 1. Single generic GPU partition
#SBATCH --partition=gpu

# 2. One generic GPU
#SBATCH --gres=gpu:1

# 3. Exact feature names with underscores (NO QUOTES)
#SBATCH --constraint=rtx_3090|rtx_4090|rtx_6000|rtx_pro_6000

# 4. Node exclusion -- DELIBERATELY NOT AN #SBATCH DIRECTIVE.
#
#    Slurm validates --exclude at submission time and rejects the entire job if
#    even one name is unknown to the controller:
#        sbatch: error: Batch job submission failed: Invalid node name specified
#    Hard-coding a long list here makes every submission hostage to the list
#    staying in sync with the cluster. Use the wrapper instead:
#
#        sbatch/submit.sh cifar10 warmup
#
#    which intersects the wanted list with the nodes that actually exist and
#    passes only those via --exclude on the command line. Run
#    `sbatch/submit.sh --check` to see which names are stale.
#
#    The exclusion is belt-and-braces anyway: --constraint above already pins
#    the job to the four wanted GPU types, so plain `sbatch` works fine too.
#    If you do want it inline, uncomment ONE line (an #SBATCH directive must be
#    a single physical line -- Slurm stops reading at the newline):
# #SBATCH --exclude=cs-1080-[01-05],cs-2080-[01-05],cs-cpu-[05-11],cs-cpu256-01,cs-pheno-[01-12],dt-2080-[01-19],ee-l40s-[01-02],ise-1080-01,ise-2080-[01-03],ise-cpu-intl-[01-28],ise-cpu128-[01-09],ise-cpu128-[11-14],ise-cpu256-[01-32],ise-pheno-[01-12]

# =============================================================================
# Graph Flow Matching -- CIFAR-10 / ImageNet / ImageNet-LT
#
# Submit from the REPO ROOT; every path below is relative to the submit dir.
#
#   sbatch sbatch/run_gfm.sbatch <dataset> [phase] [extra run_pipeline args...]
#
# Every job renames itself to gfm-<dataset>-<phase> (gfm-inlt-erte,
# gfm-in1k-trn, gfm-c10-all), so squeue can tell concurrent runs apart. Widen
# the NAME column, which defaults to 8 characters:
#   squeue -u $USER -o '%.10i %.20j %.2t %.11M %.6D %R'
#
#   dataset : cifar10 | imagenet | imagenet-lt
#   phase   : all (default) | warmup | encode | reference | train | evaluate
#             | plot | paper | report | sweep
#             (also any comma-separated subset, e.g. encode,reference,train)
#
# Examples
#   sbatch sbatch/run_gfm.sbatch cifar10 warmup          # cache weights first
#   sbatch sbatch/run_gfm.sbatch cifar10
#   sbatch sbatch/run_gfm.sbatch imagenet-lt encode
#   sbatch --time=72:00:00 sbatch/run_gfm.sbatch imagenet all
#   sbatch sbatch/run_gfm.sbatch cifar10 sweep           # graph-correction ablation
#   sbatch sbatch/run_gfm.sbatch cifar10 train --flow_epochs 400 --lr 1e-4
#
# RUNNING DATASETS IN PARALLEL
#   The three datasets share nothing that matters: encoded latents, reference
#   sets, checkpoints and results are all keyed by dataset. Submit them at the
#   same time -- but stop before the aggregation stages, which are global and
#   would overwrite each other:
#
#     sbatch sbatch/run_gfm.sbatch cifar10     warmup     # ONCE, first
#     sbatch --time=36:00:00 sbatch/run_gfm.sbatch cifar10     encode,reference,train,evaluate
#     sbatch --time=72:00:00 sbatch/run_gfm.sbatch imagenet-lt encode,reference,train,evaluate
#     EPOCHS=25 sbatch --time=96:00:00 sbatch/run_gfm.sbatch imagenet encode,reference,train,evaluate
#     # ...then once everything has finished:
#     sbatch sbatch/run_gfm.sbatch all report        # covers ALL datasets
#
#   The 'report' phase scans WORK_DIR and rebuilds the figures and tables from
#   every run it finds, so unfinished datasets simply show as '--' until they
#   land. It needs no GPU, so it can also be run straight on a login node:
#       bash sbatch/run_gfm.sbatch all report
#
#   Run 'warmup' first and let it finish: parallel cold starts would otherwise
#   race each other downloading into the same HF_HOME/TORCH_HOME.
#
#   Do NOT run the plot/paper/report stages concurrently. They are global, not
#   per-dataset: plot_training_curves.py always writes loss_curve/loss_terms/
#   fid_curve under the same names, and build_paper.py rewrites results_table.tex
#   with only the runs it was handed -- so two concurrent 'all' jobs would each
#   erase the other's rows.
#
# Overridable via the environment:
#   DATA_ROOT_CIFAR10, DATA_ROOT_IMAGENET, WORK_DIR, HF_HOME, TORCH_HOME,
#   BATCH_SIZE, MICRO_BATCH, EPOCHS, EVAL_SAMPLES, REF_IMAGES, GFM_VARIANTS,
#   IMAGENET_LT_SPLIT, GFM_CACHE_DIR, GFM_ALLOW_PARETO
#
# -----------------------------------------------------------------------------
# GPU MEMORY -- why BATCH_SIZE is not what has to fit.
#
# With the graph correction on, the diffusion term runs attention over the whole
# 32x32 latent grid, so each attention block holds a [batch, 1024, 1024] map for
# the backward pass. Measured: batch 64 wants ~24 GB, so it dies on a 3090/4090
# (23.5 GiB usable) with
#     torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 MiB
# a few steps in. That is true of every dataset here -- at --image_size 256 they
# all encode to 4x32x32 latents, so the cost per sample is identical.
#
# So the batch is split in two:
#   BATCH_SIZE  (default 64) is the EFFECTIVE batch -- the one the optimizer
#               sees, the number the paper reports, the number to keep fixed.
#   MICRO_BATCH (default auto) is what a single forward/backward holds, i.e.
#               what has to fit in VRAM. train.py accumulates
#               BATCH_SIZE / MICRO_BATCH of them per optimizer step, so the
#               gradient -- and the LR schedule, checkpoint and FID cadences,
#               which all count optimizer steps -- are unchanged.
#
# 'auto' reads the card's memory and picks: 64 on >=40 GB, 32 on 24 GB, 16 on
# 16 GB, 8 on 12 GB. Override either one explicitly, e.g. for a 24 GB card:
#     MICRO_BATCH=32 sbatch sbatch/run_gfm.sbatch imagenet
# Accumulating costs a little throughput (the GPU is fed in smaller pieces),
# perhaps 5-15%, and nothing in accuracy.
# -----------------------------------------------------------------------------
# WALL-CLOCK BUDGET -- read this before submitting.
#
# The default --time=24:00:00 above is NOT enough for a 200-epoch run on any of
# these datasets. Training is pure fp32 (no AMP anywhere in train.py) at ~8.8
# TFLOP per step for the default DiT-B/2 reaction network at batch 64.
#
#   steps/epoch at batch 64:  CIFAR-10 781 | ImageNet-LT 1,810 | ImageNet 20,018
#
# Rough training time for 200 epochs (wide bands: measure yours, see below).
# These assume the whole effective batch fits at once; a card that has to
# accumulate (see GPU MEMORY above) is a further ~5-15% slower:
#
#   dataset        rtx_3090        rtx_4090 / 6000     rtx_pro_6000
#   CIFAR-10       ~28 h           ~13 h               ~9 h
#   ImageNet-LT    ~2.7 days       ~30 h               ~20 h
#   ImageNet       ~30 days        ~14 days            ~9 days
#
# Non-training stages, once each:
#   encode     CIFAR-10 ~10 min | ImageNet-LT ~30 min | ImageNet 2-4 h (CPU-bound
#              on JPEG decode; the latents are ~0.4/0.9/10.5 GB on disk)
#   reference  ~10-30 min        evaluate  ~20-35 min (10k samples, 12 NFE each)
#
# Suggested submissions:
#   sbatch --time=36:00:00 sbatch/run_gfm.sbatch cifar10
#   sbatch --time=72:00:00 sbatch/run_gfm.sbatch imagenet-lt
#   ImageNet at 200 epochs does not fit any sane allocation -- cut EPOCHS
#   (EPOCHS=25 -> ~1-4 days) or use --subset_frac.
#
# MEASURE INSTEAD OF GUESSING: every run writes per-step wall times to
# train_log_steps.csv. After a ~200-step pilot:
#   python scripts/estimate_runtime.py --run work/runs/cifar10/models \
#       --dataset cifar10 --epochs 200 --batch_size 64
#
# If a job does hit the limit, just resubmit the same command: train.py writes
# train_state.pt (model + optimizer + LR schedule + epoch + step) every
# --state_every_steps and at every epoch boundary, and --resume auto (the
# default) continues from it. At most one epoch of work is repeated.
#
# When the run is longer than the cluster's wall-clock cap, chain it:
#     bash sbatch/chain.sh 17 --time=04:00:00 imagenet-lt train
# which submits 17 jobs, each depending (afterany) on the previous one. Check
# the cap first -- one long job is always better:
#     sinfo -o '%20P %10l %10L'                 # partition MaxTime / DefaultTime
#     sacctmgr show qos format=Name,MaxWall     # QOS caps
# (--retrain_flow_network is NOT the way to continue: it reloads weights only,
#  restarting the optimizer moments and the cosine schedule on every link.)
# -----------------------------------------------------------------------------
# =============================================================================

set -euo pipefail

DATASET="${1:-cifar10}"
shift || true
PHASE="${1:-all}"
shift || true

case "$DATASET" in
    cifar10|imagenet|imagenet-lt) ;;
    all|any)
        # The report/plot/paper phases scan WORK_DIR and cover every dataset at
        # once, so naming a single one is misleading. Accept 'all' for them.
        case "$PHASE" in
            report|plot|paper) ;;
            *) echo "ERROR: dataset 'all' is only valid for the report/plot/paper" >&2
               echo "       phases (they aggregate every run under WORK_DIR)." >&2
               echo "       For '$PHASE', name one of: cifar10|imagenet|imagenet-lt" >&2
               exit 2 ;;
        esac ;;
    *) echo "ERROR: unknown dataset '$DATASET' (cifar10|imagenet|imagenet-lt|all)" >&2
       exit 2 ;;
esac

mkdir -p sbatch/logs

# --- job name -----------------------------------------------------------------
# The #SBATCH --job-name above is a placeholder: the dataset and phase are
# positional arguments, which Slurm has not seen when it reads the directives.
# So build the real name here and rename the job through scontrol -- squeue
# shows "gfm-inlt-enc" instead of five identical "gfm" rows.
#
# Submitting through sbatch/submit.sh sets the name at SUBMISSION time instead
# (-J), which is better: it is already right while the job is PENDING, and the
# log files pick it up too, since Slurm expands %x when it opens them.
#
# NOTE: squeue truncates NAME to 8 characters in its default format. Widen it:
#     squeue -u $USER -o '%.10i %.20j %.2t %.11M %.6D %R'
#     alias sq='squeue -u $USER -o "%.10i %.20j %.2t %.11M %.6D %R"'
case "$DATASET" in
    cifar10)     _ds=c10 ;;
    imagenet)    _ds=in1k ;;
    imagenet-lt) _ds=inlt ;;
    *)           _ds="$DATASET" ;;
esac
# One phase -> a 3-letter abbreviation; several -> their initials (encode,
# reference,train,evaluate becomes 'erte'), which stays inside squeue's column.
case "$PHASE" in
    encode)    _ph=enc ;;
    reference) _ph=ref ;;
    train)     _ph=trn ;;
    evaluate)  _ph=evl ;;
    warmup)    _ph=wrm ;;
    report)    _ph=rpt ;;
    all)       _ph=all ;;
    sweep)     _ph=swp ;;
    *)         _ph="$(echo "$PHASE" | tr ',' '\n' | cut -c1 | tr -d '\n')" ;;
esac
JOB_LABEL="gfm-${_ds}-${_ph}"

# Do not fight an explicit -J: if the name is neither the placeholder nor one
# we generated, the submitter chose it deliberately.
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    case "${SLURM_JOB_NAME:-gfm}" in
        gfm|run_gfm.sbatch|gfm-*)
            if scontrol update "JobId=$SLURM_JOB_ID" "Name=$JOB_LABEL" 2>/dev/null; then
                echo "[slurm] job name: ${JOB_LABEL}  (${DATASET} / ${PHASE})"
            fi ;;
        *) echo "[slurm] keeping explicit job name: ${SLURM_JOB_NAME}" ;;
    esac
fi


# --- environment: ACTIVATE YOUR VENV -------------------
# (If Python is not found, uncomment the module load line too)
# module load cuda/12.4 python/3.9
source .venv/bin/activate
# -------------------------------------------------------

# Per-site paths (ImageNet root, scratch, ...). Sourced before anything else so
# an explicit VAR=... on the sbatch command line still overrides it, since the
# file only sets values that are not already in the environment.
if [ -f sbatch/site.env ]; then
    echo "[setup] sourcing sbatch/site.env"
    set -a
    # shellcheck disable=SC1091
    . sbatch/site.env
    set +a
fi

# Keep model downloads off $HOME -- home quotas are usually far too small for
# the SD-VAE (~330 MB) plus the Inception weights (~105 MB), and a quota error
# mid-run looks like an unrelated crash.
export WORK_DIR="${WORK_DIR:-$PWD/work}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$WORK_DIR/torch_cache}"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# Dataset roots. CIFAR-10 downloads itself; ImageNet must already be on disk
# as <root>/train/n01440764/... and <root>/val/...
DATA_ROOT_CIFAR10="${DATA_ROOT_CIFAR10:-$WORK_DIR/data/cifar10}"
# No default: ImageNet lives somewhere different on every cluster and is
# never downloaded automatically. A wrong placeholder here just produces a
# confusing PermissionError, so require it to be set.
DATA_ROOT_IMAGENET="${DATA_ROOT_IMAGENET:-}"

# Which stages actually open the raw image tree? Only the two that walk it:
# encode (JPEG -> latents) and reference (JPEG -> the FID reference set). train
# reads the encoded latents, evaluate reads the reference set, and plot/paper/
# report read run directories -- none of them touch DATA_ROOT, so none of them
# should be blocked on a path they never open.
NEEDS_RAW=0
case ",$PHASE," in
    *,encode,*|*,reference,*|,all,|,sweep,) NEEDS_RAW=1 ;;
esac

SLUG="${DATASET//-/_}"
# Where an explicit root is not given, two places remember one that worked:
#   1. the memo this script writes after a successful check;
#   2. "source_root" in the metadata.json that encode_new_datasets.py writes
#      beside the latents -- so a tree that was encoded once stays findable even
#      if the memo predates this script or WORK_DIR was rebuilt.
IMAGENET_ROOT_MEMO="$WORK_DIR/.imagenet_root"
if [ -z "$DATA_ROOT_IMAGENET" ] && [ -r "$IMAGENET_ROOT_MEMO" ]; then
    DATA_ROOT_IMAGENET="$(cat "$IMAGENET_ROOT_MEMO")"
    echo "[data] DATA_ROOT_IMAGENET unset -> remembered root: $DATA_ROOT_IMAGENET"
fi
if [ -z "$DATA_ROOT_IMAGENET" ] && [ "$DATASET" != "cifar10" ]; then
    # imagenet and imagenet-lt are the SAME image tree (LT is a 115k subset of
    # it), so either one's encode metadata locates the root for the other. The
    # dataset's own directory is tried first, then any other encoded set; the
    # "dataset" field is checked so a CIFAR-10 encode never answers here.
    IMAGENET_ROOT_SEARCHED="$WORK_DIR/encoded/*/metadata.json"
    for _meta in "$WORK_DIR"/encoded/"$SLUG"_*/metadata.json                  "$WORK_DIR"/encoded/*/metadata.json; do
        [ -r "$_meta" ] || continue
        _root="$(python -c 'import json, sys
meta = json.load(open(sys.argv[1]))
name = str(meta.get("dataset", ""))
print(meta.get("source_root", "") if name.startswith("imagenet") else "")' "$_meta" 2>/dev/null || true)"
        if [ -n "$_root" ] && [ -d "$_root" ]; then
            DATA_ROOT_IMAGENET="$_root"
            echo "[data] recovered ImageNet root from $_meta: $DATA_ROOT_IMAGENET"
            break
        fi
    done
fi
if [ "$DATASET" == "cifar10" ]; then
    DATA_ROOT="$DATA_ROOT_CIFAR10"
else
    DATA_ROOT="$DATA_ROOT_IMAGENET"
fi

# ImageNet is never downloaded automatically. Check the path before doing
# anything expensive -- otherwise the failure surfaces as a bare PermissionError
# from inside a directory walk, after the job has already queued and started.
if [ "$DATASET" != "cifar10" ] && [ "$NEEDS_RAW" -eq 1 ]; then
    if [ ! -d "$DATA_ROOT" ] || [ ! -r "$DATA_ROOT" ] || [ ! -x "$DATA_ROOT" ]; then
        if [ -z "$DATA_ROOT" ]; then
            echo "ERROR: DATA_ROOT_IMAGENET is not set, and phase '$PHASE' reads the" >&2
            echo "       raw ImageNet images (encode and reference do; train," >&2
            echo "       evaluate, plot, paper and report do not)." >&2
            echo "       Looked for a remembered one in:" >&2
            echo "         $IMAGENET_ROOT_MEMO" >&2
            echo "         ${IMAGENET_ROOT_SEARCHED:-$WORK_DIR/encoded/*/metadata.json}  (\"source_root\")" >&2
        else
            echo "ERROR: ImageNet root not readable: $DATA_ROOT" >&2
        fi
        echo >&2
        echo "       If the latents and the reference set already exist, skip those" >&2
        echo "       two stages and no root is needed at all:" >&2
        echo "         sbatch --time=72:00:00 sbatch/run_gfm.sbatch $DATASET train,evaluate" >&2
        echo >&2
        echo "       Otherwise point it at the real location (it is then remembered" >&2
        echo "       in $IMAGENET_ROOT_MEMO for later submissions):" >&2
        echo "         DATA_ROOT_IMAGENET=/path/to/imagenet \\" >&2
        echo "           sbatch --time=72:00:00 sbatch/run_gfm.sbatch $DATASET $PHASE" >&2
        echo "       or put it in sbatch/site.env once (see sbatch/site.env.example)." >&2
        echo >&2
        echo "       It must hold EITHER the 1000 wnid folders (directly or under" >&2
        echo "       train/) OR HuggingFace parquet shards (*.parquet, directly or" >&2
        echo "       under data/).  To hunt for an existing copy on the cluster:" >&2
        echo "         find / -maxdepth 5 -name 'train-00000-of-*.parquet' 2>/dev/null | head" >&2
        echo "         ls -d /datasets /data /scratch /storage /mnt/* 2>/dev/null" >&2
        echo "         find / -maxdepth 4 -type d -name 'n01440764' 2>/dev/null | head" >&2
        echo "       (n01440764 is the first ImageNet class; finding it locates the tree.)" >&2
        exit 1
    fi
    echo "[data] ImageNet root OK: $DATA_ROOT"
    printf '%s' "$DATA_ROOT" > "$IMAGENET_ROOT_MEMO" 2>/dev/null || true
elif [ "$DATASET" != "cifar10" ]; then
    echo "[data] phase '$PHASE' does not read raw images -- ImageNet root not required"
    echo "       (train uses $WORK_DIR/encoded/${SLUG}_*, evaluate uses"
    echo "        $WORK_DIR/reference/${SLUG}_*; build those with the encode and"
    echo "        reference stages, which do need the root)"
fi

# Writable cache for the ImageNet-LT split file -- never the (read-only) dataset root.
export GFM_CACHE_DIR="${GFM_CACHE_DIR:-$WORK_DIR/cache}"
mkdir -p "$GFM_CACHE_DIR"

# ImageNet-LT split list. Prefer a local copy: without one the loader tries to
# download it, and on a compute node with no outbound network that quietly falls
# back to a Pareto *reconstruction* -- same count profile, different file list,
# so the numbers would not be comparable to published ImageNet-LT results.
# Set IMAGENET_LT_SPLIT to override; otherwise look in the usual places.
LT_EXTRA=()
if [ "$DATASET" == "imagenet-lt" ] && [ "$NEEDS_RAW" -eq 1 ]; then
    if [ -z "${IMAGENET_LT_SPLIT:-}" ]; then
        for c in "$DATA_ROOT/ImageNet_LT/ImageNet_LT_train.txt" \
                 "$DATA_ROOT/ImageNet_LT_train.txt" \
                 "$DATA_ROOT/../ImageNet_LT/ImageNet_LT_train.txt" \
                 "$DATA_ROOT/splits/ImageNet_LT_train.txt"; do
            if [ -r "$c" ]; then IMAGENET_LT_SPLIT="$c"; break; fi
        done
    fi
    if [ -n "${IMAGENET_LT_SPLIT:-}" ] && [ -r "$IMAGENET_LT_SPLIT" ]; then
        echo "[data] ImageNet-LT split (official): $IMAGENET_LT_SPLIT"
        # --no_pareto_fallback: with a real split file in hand, a silent
        # reconstruction can only mean something went wrong. Fail instead.
        LT_EXTRA=(--imagenet_lt_split_file "$IMAGENET_LT_SPLIT" --no_pareto_fallback)
        # Escape hatch for the one case where the guard is wrong: parquet shards
        # that do not carry the original ImageNet filenames, so the official
        # split cannot be matched onto them however hard we try. Opting in is
        # deliberate -- the resulting numbers are NOT comparable to published
        # ImageNet-LT results, and the run records that in its metadata.
        if [ "${GFM_ALLOW_PARETO:-0}" == "1" ]; then
            LT_EXTRA=(--imagenet_lt_split_file "$IMAGENET_LT_SPLIT")
            echo "[data] WARNING: GFM_ALLOW_PARETO=1 -- a reconstructed split is" >&2
            echo "       allowed. Same long-tailed profile, different image list;" >&2
            echo "       results are not comparable to published ImageNet-LT." >&2
        fi
    else
        echo "[data] WARNING: no local ImageNet_LT_train.txt found." >&2
        echo "       The loader will try to download it; if this node has no" >&2
        echo "       network it falls back to a Pareto RECONSTRUCTION, which is" >&2
        echo "       not the official split. Set IMAGENET_LT_SPLIT=/path/to/" >&2
        echo "       ImageNet_LT_train.txt to be sure." >&2
    fi
fi

# --- what card did we actually get? ------------------------------------------
# Read once, used twice: the micro-batch heuristic wants the memory figure, and
# the CUDA preflight below wants the name and driver version to tell "no GPU in
# this allocation" apart from "GPU present but CUDA will not start on it".
# nvidia-smi reports only the GPUs allocated to this job, so head -n1 is the
# card we will train on. '|| true': with pipefail a missing nvidia-smi (login
# node, CPU partition) would otherwise take the script down through set -e.
GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '\r' || true)"
GPU_NAME="$(printf '%s' "$GPU_INFO" | cut -d, -f1 | sed 's/^[[:space:]]*//')"
GPU_DRIVER="$(printf '%s' "$GPU_INFO" | cut -d, -f2 | tr -d '[:space:]')"
GPU_MEM_MIB="$(printf '%s' "$GPU_INFO" | cut -d, -f3 | tr -d '[:space:]')"
if [ -n "$GPU_INFO" ]; then
    echo "[gpu] ${GPU_NAME}, driver ${GPU_DRIVER}, ${GPU_MEM_MIB} MiB"
else
    echo "[gpu] nvidia-smi reported nothing (no GPU in this allocation, or no driver)"
fi

# Effective batch (what the optimizer sees) and the micro-batch that actually
# has to fit in VRAM. See the GPU MEMORY section at the top.
BATCH_SIZE="${BATCH_SIZE:-64}"
MICRO_BATCH="${MICRO_BATCH:-auto}"

if [ "$MICRO_BATCH" == "auto" ]; then
    case "${GPU_MEM_MIB:-}" in
        ''|*[!0-9]*)
            echo "[gpu] no usable memory reading -> keeping the batch whole"
            MICRO_BATCH="$BATCH_SIZE" ;;
        *)
            # ~2.5 GB fixed (weights, grads, AdamW moments) + ~0.33 GB per
            # sample of activations, against ~92% of the card.
            if   [ "$GPU_MEM_MIB" -ge 40000 ]; then MICRO_BATCH=64
            elif [ "$GPU_MEM_MIB" -ge 22000 ]; then MICRO_BATCH=32
            elif [ "$GPU_MEM_MIB" -ge 14000 ]; then MICRO_BATCH=16
            elif [ "$GPU_MEM_MIB" -ge 10000 ]; then MICRO_BATCH=8
            else                                    MICRO_BATCH=4
            fi
            echo "[gpu] ${GPU_MEM_MIB} MiB visible -> micro-batch ${MICRO_BATCH}" ;;
    esac
fi

for _v in BATCH_SIZE MICRO_BATCH; do
    case "${!_v}" in
        ''|*[!0-9]*) echo "ERROR: $_v must be a positive integer (got '${!_v}')" >&2
                     exit 2 ;;
    esac
done

# Never larger than the effective batch, and always a divisor of it so that
# micro x accum lands exactly on BATCH_SIZE.
if [ "$MICRO_BATCH" -gt "$BATCH_SIZE" ]; then MICRO_BATCH="$BATCH_SIZE"; fi
if [ "$MICRO_BATCH" -lt 1 ]; then MICRO_BATCH=1; fi
while [ $((BATCH_SIZE % MICRO_BATCH)) -ne 0 ]; do
    MICRO_BATCH=$((MICRO_BATCH - 1))
done
GRAD_ACCUM=$((BATCH_SIZE / MICRO_BATCH))

# Accumulation leaves a lot of short-lived allocations behind; expandable
# segments keep them from fragmenting the pool into an OOM that has the memory
# but not one contiguous block of it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EPOCHS="${EPOCHS:-200}"
EVAL_SAMPLES="${EVAL_SAMPLES:-10000}"
REF_IMAGES="${REF_IMAGES:-50000}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

# Slurm remaps the allocated GPU to index 0 inside the job regardless of which
# physical card it is, so cuda:0 is always correct here. train.py's upstream
# default is cuda:1, which would abort on the device check -- do not remove.
DEVICE="cuda:0"

# Fail loudly and early rather than 40 minutes into a queue slot.
# warmup (downloads weights) and report/plot/paper (pure CPU aggregation) are
# allowed to run somewhere without a GPU, e.g. a login node. Everything else
# needs CUDA.
GFM_REQUIRE_CUDA=1
case "$PHASE" in warmup|report|plot|paper) GFM_REQUIRE_CUDA=0 ;; esac
export GFM_REQUIRE_CUDA

# The banner goes BEFORE the preflight: when the preflight is what fails, the
# log still has to say which node and which card it failed on.
echo "=================================================="
echo " host=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo " dataset=${DATASET}  phase=${PHASE}  extra_args=$*"
echo " data_root=${DATA_ROOT}"
echo " work_dir=${WORK_DIR}"
echo " workers=${NUM_WORKERS}  epochs=${EPOCHS}"
echo " batch=${BATCH_SIZE} (micro ${MICRO_BATCH} x accum ${GRAD_ACCUM})"
echo " gpu=${GPU_INFO:-none}"
echo "=================================================="

if ! python - <<'PY'
import importlib.util, os, sys  # NB: bare "import importlib" hides .util
missing = [m for m in ("torch", "torchvision", "diffusers", "timm", "einops",
                       "omegaconf", "torchdiffeq", "cleanfid", "scipy",
                       "sklearn", "matplotlib", "tqdm", "pyarrow")
           if importlib.util.find_spec(m) is None]
if missing:
    print(f"ERROR: not importable: {', '.join(missing)}", file=sys.stderr)
    print("       pip install -r requirements.txt "
          "--extra-index-url https://download.pytorch.org/whl/cu124", file=sys.stderr)
    sys.exit(1)
import torch
have = torch.cuda.is_available()
if os.environ.get("GFM_REQUIRE_CUDA") == "1" and not have:
    # Two very different failures hide behind "no CUDA device". Say which one:
    # a CPU-only wheel is ours to fix, a driver that refuses to initialise is
    # the node's -- and the shell below adds the evidence either way.
    if torch.version.cuda is None:
        print(f"ERROR: torch {torch.__version__} is a CPU-ONLY build "
              f"(torch.version.cuda is None).", file=sys.stderr)
        print("       Reinstall with the --extra-index-url above.", file=sys.stderr)
    else:
        print(f"ERROR: torch {torch.__version__} is built for CUDA "
              f"{torch.version.cuda} but cannot initialise any GPU.",
              file=sys.stderr)
        print("       The wheel is fine; CUDA itself refused to start.",
              file=sys.stderr)
    sys.exit(1)
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"devices {torch.cuda.device_count() if have else 0}")
PY
then
    NODE="${SLURMD_NODENAME:-$(hostname)}"
    {
        echo "--- CUDA diagnostics -------------------------------------------"
        echo " node=${NODE}  job=${SLURM_JOB_ID:-none}"
        echo " CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}'"
        echo " SLURM_JOB_GPUS='${SLURM_JOB_GPUS:-<unset>}'  GPU_DEVICE_ORDINAL='${GPU_DEVICE_ORDINAL:-<unset>}'"
        echo " nvidia-smi: ${GPU_INFO:-<nothing>}"
        echo " /dev nodes: $(ls -1 /dev/nvidia* 2>/dev/null | tr '\n' ' ' || true)"
        if [ -n "$GPU_INFO" ]; then
            cat <<MSG
 nvidia-smi sees the card, so it is there -- but CUDA will not initialise on it
 ("CUDA unknown error", "Setting the available devices to be zero"). That is a
 node-level fault, not a problem with this repo or the venv: a wedged driver, a
 missing /dev/nvidia-uvm, or a driver too old for the cu124 wheels (torch 2.6
 +cu124 needs driver >= 525; this node reports ${GPU_DRIVER:-unknown}).
 Nothing to fix in the job. Resubmit somewhere else:
     sbatch --exclude=${NODE} sbatch/run_gfm.sbatch ${DATASET} ${PHASE}
 and add ${NODE} to WANT_EXCLUDE in sbatch/submit.sh so it stays out of the pool.
MSG
        else
            cat <<MSG
 No GPU is visible at all, so the allocation never had one. Check that the
 submission kept --gres=gpu:1 and --partition=gpu (both are #SBATCH directives
 in this script, so plain 'sbatch sbatch/run_gfm.sbatch ...' has them), and that
 the node really is a GPU node: scontrol show node ${NODE}
MSG
        fi
        echo "----------------------------------------------------------------"
    } >&2
    exit 1
fi

COMMON=(
    --dataset "$DATASET"
    --data_root "$DATA_ROOT"
    ${LT_EXTRA[@]+"${LT_EXTRA[@]}"}
    --work_dir "$WORK_DIR"
    --device "$DEVICE"
    --num_workers "$NUM_WORKERS"
    --train_batch_size "$MICRO_BATCH"
    --grad_accum_steps "$GRAD_ACCUM"
    --flow_epochs "$EPOCHS"
    --num_eval_samples "$EVAL_SAMPLES"
    --num_reference_images "$REF_IMAGES"
)

# --- EXECUTION ---

if [ "$PHASE" == "warmup" ]; then
    # Pull the pretrained weights into the caches above. Run this once on a node
    # with outbound network access; compute nodes on many clusters have none,
    # and the first real job would otherwise die at the VAE download.
    echo ">>> caching SD-VAE and Inception weights into $HF_HOME / $TORCH_HOME"
    python - <<'PY'
import torch
from diffusers import AutoencoderKL
AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
print("sd-vae-ft-mse cached")
from torchvision.models import inception_v3, Inception_V3_Weights
inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
print("inception_v3 cached")
from cleanfid.features import build_feature_extractor
# build_feature_extractor defaults to device('cuda'); pass it explicitly so
# warmup also works on a login node that has network access but no GPU.
build_feature_extractor(
    "clean",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_dataparallel=False)
print("clean-fid inception cached")
PY
    echo "warmup complete"

elif [ "$PHASE" == "report" ]; then
    # Aggregate EVERY finished run under $WORK_DIR into one figure set and one
    # set of LaTeX tables.
    #
    # This exists because `plot` and `paper` are global, not per-dataset:
    # plot_training_curves.py always writes loss_curve/loss_terms/fid_curve
    # (same filenames whatever the dataset) and build_paper.py rewrites
    # results_table.tex containing only the runs handed to it. Two dataset jobs
    # running those stages concurrently would overwrite each other's output.
    # So: run the per-dataset stages in parallel, then this once at the end.
    echo "=================================================="
    echo " COLLECTING RESULTS from ${WORK_DIR}"
    echo "=================================================="
    PLOT_ARGS=()
    PAPER_ARGS=()
    n_found=0
    for d in "$WORK_DIR"/runs/*/models; do
        [ -d "$d" ] || continue
        v="$(basename "$(dirname "$d")")"
        label="${v//__/-}"
        res="$WORK_DIR/results/$v"
        have_log=0; [ -s "$d/train_log_steps.csv" ] && have_log=1
        have_res=0; [ -s "$res/metrics.json" ] && have_res=1
        if [ "$have_log" -eq 0 ] && [ "$have_res" -eq 0 ]; then
            continue
        fi
        echo "  ${label}: train-log=${have_log} metrics=${have_res}"
        if [ "$have_log" -eq 1 ]; then
            PLOT_ARGS+=(--run "${label}=${d}")
        fi
        PAPER_ARGS+=(--run "${label}=${d}")
        if [ "$have_res" -eq 1 ]; then
            PAPER_ARGS+=(--result "${label}=${res}")
        fi
        n_found=$((n_found + 1))
    done

    if [ "$n_found" -eq 0 ]; then
        echo "nothing to report yet under ${WORK_DIR}/runs/*/models"
        exit 0
    fi
    if [ "${#PLOT_ARGS[@]}" -gt 0 ]; then
        python scripts/plot_training_curves.py "${PLOT_ARGS[@]}" --out_dir paper/figures
    fi
    python scripts/build_paper.py "${PAPER_ARGS[@]}"
    echo "collected ${n_found} run(s) -- build the PDF with: cd paper && latexmk -pdf main.tex"

elif [ "$PHASE" == "sweep" ]; then
    # Ablation of the graph correction -- the term the GFM paper adds. The
    # encode and reference stages are dataset-level, so they run once; --tag
    # then keeps each variant's checkpoints and metrics in its own directory
    # while sharing those two expensive stages.
    echo "=================================================="
    echo " GRAPH-CORRECTION ABLATION SWEEP on ${DATASET}"
    echo "=================================================="

    python scripts/run_pipeline.py "${COMMON[@]}" --stages encode,reference "$@"

    # name:pipeline-flags. gaussian is deliberately absent: its kernel
    # underflows to zero on these latents and training NaNs immediately.
    #
    # These run SERIALLY in one job, so the full sweep costs 4x a single
    # training -- days, not hours. To spread it over four jobs instead, set
    # GFM_VARIANTS to one entry per submission:
    #
    #   for v in "graph_attention:--adj_mode attention" \
    #            "graph_cosine:--adj_mode cosine" \
    #            "graph_knn:--adj_mode knn" \
    #            "no_graph:--no_diffusion"; do
    #     GFM_VARIANTS="$v" sbatch --time=36:00:00 sbatch/run_gfm.sbatch cifar10 sweep
    #   done
    #
    # (encode/reference are idempotent, so re-running them per job is cheap; the
    #  final paper build then reports whichever variants have finished.)
    if [ -n "${GFM_VARIANTS:-}" ]; then
        IFS=';' read -r -a VARIANTS <<< "$GFM_VARIANTS"
    else
        VARIANTS=(
            "graph_attention:--adj_mode attention"
            "graph_cosine:--adj_mode cosine"
            "graph_knn:--adj_mode knn"
            "no_graph:--no_diffusion"
        )
        echo "NOTE: running ${#VARIANTS[@]} variants serially in this one job."
        echo "      Budget ~${#VARIANTS[@]}x a single training, or split with"
        echo "      GFM_VARIANTS (see the comment in this script)."
    fi

    for v in "${VARIANTS[@]}"; do
        name="${v%%:*}"
        flags="${v#*:}"
        echo ">>> variant ${name}  (${flags})"
        # shellcheck disable=SC2086
        python scripts/run_pipeline.py "${COMMON[@]}" \
            --tag "$name" --stages train,evaluate $flags "$@"
        echo "finished ${name}"
        echo "--------------------------------------------------"
    done

    # One figure set and one paper build comparing every variant. Both argument
    # lists are built as arrays: the labels contain no spaces by construction,
    # but a command substitution here would word-split any that appeared later.
    SLUG="${DATASET//-/_}"
    PAPER_ARGS=()
    PLOT_ARGS=()
    for v in "${VARIANTS[@]}"; do
        name="${v%%:*}"
        label="${DATASET}-${name}"
        PAPER_ARGS+=(--result "${label}=${WORK_DIR}/results/${SLUG}__${name}"
                     --run    "${label}=${WORK_DIR}/runs/${SLUG}__${name}/models")
        PLOT_ARGS+=(--run "${label}=${WORK_DIR}/runs/${SLUG}__${name}/models")
    done
    python scripts/plot_training_curves.py "${PLOT_ARGS[@]}" --out_dir paper/figures
    python scripts/build_paper.py "${PAPER_ARGS[@]}"
    echo "sweep complete -- build the PDF with: cd paper && latexmk -pdf main.tex"

else
    # Standard single-phase (or full) execution.
    python scripts/run_pipeline.py "${COMMON[@]}" --stages "$([ "$PHASE" == "all" ] \
        && echo "encode,reference,train,evaluate,plot,paper" || echo "$PHASE")" "$@"
fi

echo "done: dataset=${DATASET} phase=${PHASE}"

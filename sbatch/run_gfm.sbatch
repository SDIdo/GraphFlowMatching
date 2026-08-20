#!/bin/bash
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

# 4. Exclude the non-matching nodes.
#    NOTE: an #SBATCH directive must be ONE physical line -- Slurm stops
#    reading at the newline, so a wrapped list silently truncates the exclusion.
#SBATCH --exclude=cs-1080-[01-05],cs-2080-[01-05],cs-cpu-[05-11],cs-cpu256-01,cs-pheno-[01-12],dt-2080-[01-19],ee-l40s-[01-02],ise-1080-01,ise-2080-[01-03],ise-cpu-intl-[01-28],ise-cpu128-[01-09],ise-cpu128-[11-14],ise-cpu256-[01-32],ise-pheno-[01-12]

# =============================================================================
# Graph Flow Matching -- CIFAR-10 / ImageNet / ImageNet-LT
#
# Submit from the REPO ROOT; every path below is relative to the submit dir.
#
#   sbatch sbatch/run_gfm.sbatch <dataset> [phase] [extra run_pipeline args...]
#
#   dataset : cifar10 | imagenet | imagenet-lt
#   phase   : all (default) | warmup | encode | reference | train | evaluate
#             | plot | paper | sweep
#
# Examples
#   sbatch sbatch/run_gfm.sbatch cifar10 warmup          # cache weights first
#   sbatch sbatch/run_gfm.sbatch cifar10
#   sbatch sbatch/run_gfm.sbatch imagenet-lt encode
#   sbatch --time=72:00:00 sbatch/run_gfm.sbatch imagenet all
#   sbatch sbatch/run_gfm.sbatch cifar10 sweep           # graph-correction ablation
#   sbatch sbatch/run_gfm.sbatch cifar10 train --flow_epochs 400 --lr 1e-4
#
# Overridable via the environment:
#   DATA_ROOT_CIFAR10, DATA_ROOT_IMAGENET, WORK_DIR, HF_HOME, TORCH_HOME,
#   BATCH_SIZE, EPOCHS, EVAL_SAMPLES, REF_IMAGES, GFM_VARIANTS
#
# -----------------------------------------------------------------------------
# WALL-CLOCK BUDGET -- read this before submitting.
#
# The default --time=24:00:00 above is NOT enough for a 200-epoch run on any of
# these datasets. Training is pure fp32 (no AMP anywhere in train.py) at ~8.8
# TFLOP per step for the default DiT-B/2 reaction network at batch 64.
#
#   steps/epoch at batch 64:  CIFAR-10 781 | ImageNet-LT 1,810 | ImageNet 20,018
#
# Rough training time for 200 epochs (wide bands: measure yours, see below):
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
# If a job does hit the limit, restart it with --retrain_flow_network. That
# reloads the weights only -- optimizer state and the cosine LR schedule restart
# -- so prefer one long job over several chained ones.
# -----------------------------------------------------------------------------
# =============================================================================

set -euo pipefail

DATASET="${1:-cifar10}"
shift || true
PHASE="${1:-all}"
shift || true

case "$DATASET" in
    cifar10|imagenet|imagenet-lt) ;;
    *) echo "ERROR: unknown dataset '$DATASET' (cifar10|imagenet|imagenet-lt)" >&2
       exit 2 ;;
esac

mkdir -p sbatch/logs

# --- environment: ACTIVATE YOUR VENV -------------------
# (If Python is not found, uncomment the module load line too)
# module load cuda/12.4 python/3.9
source .venv/bin/activate
# -------------------------------------------------------

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
DATA_ROOT_IMAGENET="${DATA_ROOT_IMAGENET:-/datasets/imagenet}"
if [ "$DATASET" == "cifar10" ]; then
    DATA_ROOT="$DATA_ROOT_CIFAR10"
else
    DATA_ROOT="$DATA_ROOT_IMAGENET"
fi

BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-200}"
EVAL_SAMPLES="${EVAL_SAMPLES:-10000}"
REF_IMAGES="${REF_IMAGES:-50000}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

# Slurm remaps the allocated GPU to index 0 inside the job regardless of which
# physical card it is, so cuda:0 is always correct here. train.py's upstream
# default is cuda:1, which would abort on the device check -- do not remove.
DEVICE="cuda:0"

# Fail loudly and early rather than 40 minutes into a queue slot.
# `warmup` only downloads weights, so it is allowed to run somewhere without a
# GPU (a login node); every other phase needs CUDA.
GFM_REQUIRE_CUDA=1
[ "$PHASE" == "warmup" ] && GFM_REQUIRE_CUDA=0
export GFM_REQUIRE_CUDA

python - <<'PY' || exit 1
import importlib.util, os, sys  # NB: bare "import importlib" hides .util
missing = [m for m in ("torch", "torchvision", "diffusers", "timm", "einops",
                       "omegaconf", "torchdiffeq", "cleanfid", "scipy",
                       "sklearn", "matplotlib", "tqdm")
           if importlib.util.find_spec(m) is None]
if missing:
    print(f"ERROR: not importable: {', '.join(missing)}", file=sys.stderr)
    print("       pip install -r requirements.txt "
          "--extra-index-url https://download.pytorch.org/whl/cu124", file=sys.stderr)
    sys.exit(1)
import torch
have = torch.cuda.is_available()
if os.environ.get("GFM_REQUIRE_CUDA") == "1" and not have:
    print(f"ERROR: torch {torch.__version__} reports no CUDA device.", file=sys.stderr)
    print("       Either the allocation has no GPU, or this is a CPU-only "
          "wheel -- reinstall with the --extra-index-url above.", file=sys.stderr)
    sys.exit(1)
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"devices {torch.cuda.device_count() if have else 0}")
PY

echo "=================================================="
echo " host=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo " dataset=${DATASET}  phase=${PHASE}  extra_args=$*"
echo " data_root=${DATA_ROOT}"
echo " work_dir=${WORK_DIR}"
echo " workers=${NUM_WORKERS}  batch=${BATCH_SIZE}  epochs=${EPOCHS}"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

COMMON=(
    --dataset "$DATASET"
    --data_root "$DATA_ROOT"
    --work_dir "$WORK_DIR"
    --device "$DEVICE"
    --num_workers "$NUM_WORKERS"
    --train_batch_size "$BATCH_SIZE"
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
    VARIANTS=(
        "graph_attention:--adj_mode attention"
        "graph_cosine:--adj_mode cosine"
        "graph_knn:--adj_mode knn"
        "no_graph:--no_diffusion"
    )

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

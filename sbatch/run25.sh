#!/bin/bash
# =============================================================================
# Submit the whole 25-epoch pipeline for cifar10, imagenet-lt and imagenet
# under a hard 4 h wall-clock cap.
#
# WHY THIS EXISTS
#   25 epochs is more than 4 h of training on every one of these datasets
#   except CIFAR-10 on a fast card, so no single job can finish one. This
#   submits, per dataset:
#
#       warmup (once, shared) --> encode[,reference] --> train x N --> evaluate
#
#   every job asking for LINK_TIME (default 03:55:00, just under the cap) and
#   every arrow an afterany dependency. The N train links are a chain: each one
#   resumes from <run_dir>/train_state.pt (model + AdamW moments + cosine
#   schedule + epoch + step), so the chain behaves like one long run. See
#   sbatch/chain.sh.
#
#   A link that starts after training already reached 25 epochs costs a couple
#   of minutes: train.py's epoch loop is range(start_epoch, epochs), so it is
#   empty, the final vel_net.pt is rewritten and the job exits. Over-provision
#   the link count rather than under-provision it -- a chain that runs out of
#   links leaves the run unfinished and needs another chain hung off the end.
#
# USAGE
#   bash sbatch/run25.sh                          # all three datasets
#   bash sbatch/run25.sh cifar10 imagenet-lt      # a subset
#   bash sbatch/run25.sh --dry-run imagenet       # print, submit nothing
#
# ENVIRONMENT (all optional)
#   EPOCHS=25            epochs per dataset
#   LINK_TIME=03:55:00   --time for every job
#   LINKS_CIFAR10=2  LINKS_IMAGENET_LT=4  LINKS_IMAGENET=28
#                        train links per dataset; see LINK COUNTS below
#   ENCODE_CPUS=32       --cpus-per-task for the ImageNet encode (CPU-bound)
#   ENCODE_MEM=96G       --mem for the ImageNet encode
#   SKIP_WARMUP=1        the HF weight cache is already populated
#   SKIP_ENCODE=1        work/encoded/<ds> and work/reference/<ds> already exist
#   EXTRA=--constraint=rtx_pro_6000   sbatch options added to every job
#
# LINK COUNTS
#   The defaults assume the SLOWEST card the job constraint allows (rtx_3090)
#   and are read off the 200-epoch bands in run_gfm.sbatch, scaled to 25 epochs
#   and padded ~15% for gradient accumulation and per-link startup:
#
#     dataset       25 ep on 3090   links @ 3h55   on rtx_pro_6000
#     CIFAR-10      ~3.5 h          2              ~1.1 h  -> 1
#     ImageNet-LT   ~8 h            4              ~2.5 h  -> 1-2
#     ImageNet      ~90 h           28             ~27 h   -> 8
#
#   28 pending jobs for ImageNet is a real queue commitment. Pinning to the
#   fast cards cuts it to a third:
#       EXTRA=--constraint=rtx_pro_6000 LINKS_IMAGENET=8 bash sbatch/run25.sh imagenet
#   and --subset_frac cuts it further. MEASURE rather than trust the table --
#   after ~200 steps of the first link:
#       python scripts/estimate_runtime.py --run work/runs/imagenet/models
#           --dataset imagenet --epochs 25 --batch_size 64
#   then resubmit the rest with the link count that number implies.
#
# THE ONE STAGE THAT IS NOT CHAINABLE
#   encode is not resumable: it writes latent shards and only then the
#   metadata.json that marks them usable, so a timed-out encode starts over.
#   ImageNet encode is 2-4 h at 8 workers, which is too close to the cap, so it
#   is submitted alone with ENCODE_CPUS cores (JPEG decode is the bottleneck,
#   so more cores is close to linear) and reference follows in its own job.
#   If it still times out, raise ENCODE_CPUS, or run encode once somewhere
#   without the cap.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=0
DATASETS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,68p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        cifar10|imagenet|imagenet-lt) DATASETS+=("$arg") ;;
        *) echo "ERROR: unknown argument '$arg' (cifar10|imagenet|imagenet-lt|--dry-run)" >&2; exit 2 ;;
    esac
done
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(cifar10 imagenet-lt imagenet)

export EPOCHS="${EPOCHS:-25}"
LINK_TIME="${LINK_TIME:-03:55:00}"
LINKS_CIFAR10="${LINKS_CIFAR10:-2}"
LINKS_IMAGENET_LT="${LINKS_IMAGENET_LT:-4}"
LINKS_IMAGENET="${LINKS_IMAGENET:-28}"
ENCODE_CPUS="${ENCODE_CPUS:-32}"
ENCODE_MEM="${ENCODE_MEM:-96G}"
SKIP_WARMUP="${SKIP_WARMUP:-0}"
SKIP_ENCODE="${SKIP_ENCODE:-0}"
EXTRA="${EXTRA:-}"

# EXTRA is a string of sbatch options; word-splitting it is the point.
# shellcheck disable=SC2206
EXTRA_OPTS=(${EXTRA})

# Submit one job through submit.sh and echo its id on stdout. Everything the
# user should see goes to stderr, so the id is the only thing captured.
submit_one() {
    local out ok=0 id
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  would submit: submit.sh $*" >&2
        echo "DRY"
        return 0
    fi
    out="$(bash sbatch/submit.sh "$@" 2>&1)" || ok=$?
    id="$(printf '%s\n' "$out" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n1)"
    if [ "$ok" -ne 0 ] || [ -z "$id" ]; then
        printf '%s\n' "$out" >&2
        echo "ERROR: submission failed (submit.sh exited ${ok}): $*" >&2
        exit 1
    fi
    echo "$id"
}

# Same, for a chain of dependent train links. Echoes the LAST link's id.
submit_chain() {
    local links="$1"; shift
    local out ok=0 id
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  would submit: chain.sh $links $*" >&2
        echo "DRY"
        return 0
    fi
    out="$(bash sbatch/chain.sh "$links" "$@" 2>&1)" || ok=$?
    printf '%s\n' "$out" | sed -n 's/^  link/    link/p' >&2
    id="$(printf '%s\n' "$out" | sed -n 's/^last link: \([0-9][0-9]*\).*/\1/p' | tail -n1)"
    if [ "$ok" -ne 0 ] || [ -z "$id" ]; then
        printf '%s\n' "$out" >&2
        echo "ERROR: chain failed (chain.sh exited ${ok})" >&2
        exit 1
    fi
    echo "$id"
}

# 'DRY' is not a job id, so a --dry-run must not emit --dependency=afterany:DRY.
dep_opt() {
    if [ -z "${1:-}" ] || [ "${1:-}" = "DRY" ]; then return 0; fi
    echo "--dependency=afterany:$1"
}

echo "=================================================="
echo " 25-epoch pipeline under a ${LINK_TIME} cap"
echo " datasets : ${DATASETS[*]}"
echo " epochs   : ${EPOCHS}"
echo " extra    : ${EXTRA:-(none)}"
[ "$DRY_RUN" -eq 1 ] && echo " DRY RUN  : nothing is submitted"
echo "=================================================="
echo

WARM=""
if [ "$SKIP_WARMUP" -eq 0 ]; then
    # One warmup for all three: the HF weight cache is shared and the dataset
    # argument only decides which config it loads. No GPU needed.
    echo "warmup (shared weight cache):"
    WARM="$(submit_one --time=00:40:00 ${EXTRA_OPTS[@]+"${EXTRA_OPTS[@]}"} cifar10 warmup)"
    echo "  job ${WARM}"
    echo
fi

EVALS=()
for ds in "${DATASETS[@]}"; do
    case "$ds" in
        cifar10)     links="$LINKS_CIFAR10" ;;
        imagenet-lt) links="$LINKS_IMAGENET_LT" ;;
        imagenet)    links="$LINKS_IMAGENET" ;;
    esac

    echo "${ds}: ${EPOCHS} epochs in ${links} train link(s) of ${LINK_TIME}"

    PREP=""
    if [ "$SKIP_ENCODE" -eq 0 ]; then
        if [ "$ds" = "imagenet" ]; then
            # 1.28M JPEGs: encode alone, with more cores, then reference.
            ENC="$(submit_one --time="$LINK_TIME" --cpus-per-task="$ENCODE_CPUS" --mem="$ENCODE_MEM" $(dep_opt "$WARM") ${EXTRA_OPTS[@]+"${EXTRA_OPTS[@]}"} "$ds" encode)"
            echo "  encode    ${ENC}  (${ENCODE_CPUS} cpus, ${ENCODE_MEM})"
            PREP="$(submit_one --time="$LINK_TIME" $(dep_opt "$ENC") ${EXTRA_OPTS[@]+"${EXTRA_OPTS[@]}"} "$ds" reference)"
            echo "  reference ${PREP}  (after ${ENC})"
        else
            PREP="$(submit_one --time="$LINK_TIME" $(dep_opt "$WARM") ${EXTRA_OPTS[@]+"${EXTRA_OPTS[@]}"} "$ds" encode,reference)"
            echo "  encode+reference ${PREP}"
        fi
    fi

    LAST="$(submit_chain "$links" --time="$LINK_TIME" $(dep_opt "${PREP:-$WARM}") ${EXTRA_OPTS[@]+"${EXTRA_OPTS[@]}"} "$ds" train)"
    echo "  train     ${links} link(s), last is ${LAST}"

    EVAL="$(submit_one --time="$LINK_TIME" $(dep_opt "$LAST") ${EXTRA_OPTS[@]+"${EXTRA_OPTS[@]}"} "$ds" evaluate)"
    echo "  evaluate  ${EVAL}  (after ${LAST})"
    EVALS+=("$EVAL")
    echo
done

# One report over every dataset, after the last evaluate. 'all report' scans
# work/ and prints '--' for anything unfinished, so it is safe even if one
# dataset fell over.
if [ "$DRY_RUN" -eq 0 ] && [ ${#EVALS[@]} -gt 0 ]; then
    DEPS="$(IFS=:; echo "${EVALS[*]}")"
    REP="$(submit_one --time=00:40:00 "--dependency=afterany:${DEPS}" all report)"
    echo "report ${REP}  (after ${DEPS})"
    echo
fi

echo "watch it with:"
echo "  squeue -u \$USER -o '%.10i %.20j %.2t %.11M %.20E %R'"
echo "if a link dies, resubmit the same command with SKIP_WARMUP=1 SKIP_ENCODE=1"
echo "-- training continues from train_state.pt."

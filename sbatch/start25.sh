#!/bin/bash
# =============================================================================
# Start the 25-epoch runs: one self-chaining job per dataset.
#
#     bash sbatch/start25.sh
#
# Submits three ordinary jobs -- gfm-c10, gfm-inlt, gfm-in1k -- each of which
# carries its own dataset all the way through: encode, reference, train to 25
# epochs across as many 3h55 links as it takes, then evaluate in the same job
# the moment training finishes. The last dataset to finish builds the report.
# See sbatch/run_all25.sbatch for how the chaining works.
#
#     bash sbatch/start25.sh cifar10 imagenet-lt   # a subset
#     bash sbatch/start25.sh --dry-run             # print, submit nothing
#     bash sbatch/start25.sh --exclude=cs-2080-01  # extra sbatch options are
#                                                  # forwarded, and inherited
#                                                  # by every later link
#
# Node exclusion goes through sbatch/submit.sh, which drops names that do not
# exist on this cluster before passing them to sbatch -- an invalid one makes
# sbatch reject the whole submission. That automatic list applies to the FIRST
# link of each chain; later links rely on the --constraint in
# run_all25.sbatch, which already narrows the pool to the four wanted card
# types. An --exclude you pass here is different: it is forwarded to every
# link, because it also goes into GFM_SBATCH_EXTRA.
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=0
DATASETS=()
PASSTHRU=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        cifar10|imagenet|imagenet-lt) DATASETS+=("$arg") ;;
        -*) PASSTHRU+=("$arg") ;;
        *) echo "ERROR: unknown argument '$arg'" >&2
           echo "usage: bash sbatch/start25.sh [--dry-run] [sbatch options] [datasets]" >&2
           exit 2 ;;
    esac
done
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(cifar10 imagenet-lt imagenet)

# Slurm opens --output BEFORE the job script runs: if this directory is
# missing the job starts and dies immediately, with no log to say why.
mkdir -p sbatch/logs

# Every link inherits this, so the report waits for exactly the datasets that
# were launched -- not for one that was never submitted.
export GFM_DATASETS="${DATASETS[*]}"
export EPOCHS="${EPOCHS:-25}"
export GFM_REPO="$PWD"
# Options that must apply to every self-submitted link, not just the first.
if [ ${#PASSTHRU[@]} -gt 0 ]; then
    export GFM_SBATCH_EXTRA="${GFM_SBATCH_EXTRA:-} ${PASSTHRU[*]}"
fi

echo "=================================================="
echo " 25-epoch runs, one self-chaining job per dataset"
echo " datasets : ${DATASETS[*]}"
echo " epochs   : ${EPOCHS}"
echo " extra    : ${GFM_SBATCH_EXTRA:-(none)}"
[ "$DRY_RUN" -eq 1 ] && echo " DRY RUN  : nothing is submitted"
echo "=================================================="
echo

IDS=()
for ds in "${DATASETS[@]}"; do
    case "$ds" in
        cifar10)     short=c10 ;;
        imagenet)    short=in1k ;;
        imagenet-lt) short=inlt ;;
    esac
    name="gfm-${short}"

    # submit.sh validates the node-exclusion list and then submits; the script
    # it submits is normally run_gfm.sbatch, so point it at this one.
    args=(--job-name="$name" ${PASSTHRU[@]+"${PASSTHRU[@]}"})
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  would submit: ${name}  (sbatch ${args[*]} sbatch/run_all25.sbatch ${ds})"
        continue
    fi
    out="$(GFM_SUBMIT_SCRIPT=sbatch/run_all25.sbatch \
           bash sbatch/submit.sh "${args[@]}" "$ds" 2>&1)"
    ok=$?
    id="$(printf '%s\n' "$out" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n1)"
    if [ "$ok" -ne 0 ] || [ -z "$id" ]; then
        printf '%s\n' "$out" >&2
        echo "ERROR: could not submit ${ds}" >&2
        exit 1
    fi
    echo "  ${name}  job ${id}  (${ds})"
    IDS+=("$id")
done

[ "$DRY_RUN" -eq 1 ] && exit 0

# A job that is not running yet is normal -- but WHY it is not running is the
# thing everyone wants to know, and squeue's REASON column answers it:
# Resources / Priority = queued behind other work, and it will start.
# BadConstraints / ReqNodeNotAvail / QOSMax* = it never will, fix and resubmit.
echo
echo "state now (REASON says why anything is still pending):"
squeue -j "$(IFS=,; echo "${IDS[*]}")" -o '%.10i %.14j %.2t %.11M %.11l %.20R' 2>/dev/null \
    || squeue -u "$USER" -o '%.10i %.14j %.2t %.11M %.20R'
echo
echo "watch it:  squeue -u \$USER -o '%.10i %.20j %.2t %.11M %.20E %R'"
echo "           tail -f sbatch/logs/gfm-*_*.out"
echo "The NAME column tracks the stage: gfm-c10-enc -> -ref -> -trn -> -evl."
echo "stop everything after the current links:  touch \${WORK_DIR:-\$PWD/work}/.gfm25/stop"

#!/bin/bash
# =============================================================================
# Continue CIFAR-10 from the 25-epoch state to 200 epochs, then evaluate it
# with a real step count.
#
#     bash sbatch/start200_cifar.sh              # archive, clear, submit
#     bash sbatch/start200_cifar.sh --dry-run    # show what it would do
#
# WHY THIS EXISTS AND start25.sh DOES NOT DO IT
#   Three pieces of state left by the 25-epoch run would quietly break a
#   re-launch, and all three are invisible until the run has burned its GPU
#   hours:
#
#   1. results/cifar10/metrics.json still exists. run_all25.sbatch decides
#      "already evaluated" with `[ -s "$RES_JSON" ]`, so the link that reaches
#      epoch 200 would skip evaluation entirely and the chain would end
#      declaring success while the only metrics on disk describe the 25-epoch
#      model. The file is archived, not deleted.
#
#   2. .gfm25/ holds done_cifar10, report.ok and the stall counters from the
#      finished run. Left in place they make the report stage think it is up to
#      date.
#
#   3. NSTEPS. The 25-epoch evaluation integrated the ODE with 3 RK4 steps,
#      which is the run_pipeline.py default. Whatever else was wrong, 3 NFE
#      cannot produce an image; this run evaluates at 50 unless NSTEPS says
#      otherwise.
#
# WHAT IS *NOT* RESET
#   train_state.pt. Training resumes at epoch 25 and runs to 200 rather than
#   starting over -- the cosine schedule is keyed to --T_cosine_scheduler
#   (303,335 steps), not to the epoch count, so 25 epochs in the LR is still
#   near 3e-4 and nothing has been annealed away. See the [resume] note in
#   train.py. Pass --scratch to train from zero instead.
#
#   The encoded latents and the reference set are also kept: they do not depend
#   on the epoch count, and re-encoding CIFAR-10 is an hour for nothing.
# =============================================================================
set -uo pipefail

# This is a launcher, not a batch script: it has no #SBATCH headers and its
# whole job is to submit the chain and exit. Run it with bash on a login node.
#
# It still has to survive `sbatch sbatch/start200_cifar.sh`, because that is
# the natural thing to type in a directory full of sbatch scripts. Under sbatch
# the script runs from a spool copy, so $0 is NOT in the repo and dirname $0
# would cd somewhere else entirely; SLURM_SUBMIT_DIR is the repo, and is used
# the same way run_all25.sbatch uses it.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    cd "${GFM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}" || exit 1
    echo "NOTE: this was submitted with sbatch. It works, but it is a launcher:"
    echo "      it burns a queue slot to spend one second calling sbatch again."
    echo "      Next time just run it on the login node:"
    echo "          bash sbatch/start200_cifar.sh"
    echo
else
    cd "$(dirname "$0")/.." || exit 1
fi

if [ ! -f train.py ] || [ ! -f sbatch/start25.sh ]; then
    echo "ERROR: $PWD is not the GraphFlowMatching repo (no train.py)." >&2
    echo "       Run this from the repo root, or set GFM_REPO=/path/to/repo." >&2
    exit 1
fi

DRY_RUN=0
SCRATCH=0
PASSTHRU=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --scratch) SCRATCH=1 ;;
        -h|--help) sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) PASSTHRU+=("$arg") ;;
    esac
done

# Resolved exactly as run_all25.sbatch resolves it, or the archive step would
# tidy up a directory no job will ever read.
if [ -f sbatch/site.env ]; then
    set -a
    # shellcheck disable=SC1091
    . sbatch/site.env
    set +a
fi
WORK_DIR="${WORK_DIR:-$PWD/work}"
STATE_DIR="$WORK_DIR/.gfm25"
RES_DIR="$WORK_DIR/results/cifar10"
RUN_DIR="$WORK_DIR/runs/cifar10/models"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$WORK_DIR/results/cifar10__25ep_nfe3_${STAMP}"

export EPOCHS="${EPOCHS:-200}"
export NSTEPS="${NSTEPS:-50}"
# 200 epochs is eight times the training of the last run, so the chain needs
# room for eight times the links before it calls itself a crash loop.
export GFM_MAX_LINKS="${GFM_MAX_LINKS:-160}"
export GFM_DATASETS="cifar10"
export GFM_REPO="$PWD"

say() { echo "  $*"; }
do_or_show() {
    if [ "$DRY_RUN" -eq 1 ]; then say "would: $*"; else eval "$@"; fi
}

echo "=================================================="
echo " CIFAR-10 -> ${EPOCHS} epochs, evaluate at ${NSTEPS} NFE"
echo " work_dir : ${WORK_DIR}"
echo " resume   : $([ "$SCRATCH" -eq 1 ] && echo 'no -- training from scratch' \
                                        || echo 'yes -- continues train_state.pt')"
[ "$DRY_RUN" -eq 1 ] && echo " DRY RUN  : nothing is moved or submitted"
echo "=================================================="
echo

echo "1. archiving the 25-epoch evaluation"
if [ -e "$RES_DIR" ]; then
    do_or_show "mv '$RES_DIR' '$ARCHIVE'"
    say "-> $ARCHIVE"
    say "   (metrics.json left in place would make the chain skip evaluation)"
else
    say "nothing at $RES_DIR -- skipping"
fi

echo
echo "2. clearing the chain bookkeeping for cifar10"
for f in done_cifar10 report.ok cifar10.stalls cifar10.gaveup cifar10.sig \
         cifar10.links cifar10.evaluate.ok stop; do
    if [ -e "$STATE_DIR/$f" ]; then
        do_or_show "rm -f '$STATE_DIR/$f'"
        say "removed $f"
    fi
done
say "kept: encode/reference stamps -- those artefacts are still valid"

echo
echo "3. training state"
if [ "$SCRATCH" -eq 1 ]; then
    if [ -e "$RUN_DIR/train_state.pt" ]; then
        do_or_show "mv '$RUN_DIR/train_state.pt' '$RUN_DIR/train_state.pt.25ep_${STAMP}'"
        say "moved train_state.pt aside -- training restarts at epoch 0"
    fi
    # The CSVs are appended to, so a fresh run must not inherit 25 epochs of
    # rows: epochs_done() reads the last line and would think it is done.
    for c in train_log_epochs.csv train_log_steps.csv train_log_fid.csv; do
        [ -e "$RUN_DIR/$c" ] && do_or_show "mv '$RUN_DIR/$c' '$RUN_DIR/${c}.25ep_${STAMP}'"
    done
else
    if [ -e "$RUN_DIR/train_state.pt" ]; then
        say "keeping train_state.pt -- resumes at epoch $(tail -n 1 \
            "$RUN_DIR/train_log_epochs.csv" 2>/dev/null | cut -d, -f1 || echo '?')"
    else
        say "no train_state.pt found -- this will train from scratch anyway"
    fi
fi

echo
echo "4. submitting"
if [ "$DRY_RUN" -eq 1 ]; then
    say "would submit: EPOCHS=${EPOCHS} NSTEPS=${NSTEPS} GFM_MAX_LINKS=${GFM_MAX_LINKS} \\"
    say "                bash sbatch/start25.sh cifar10 ${PASSTHRU[*]:-}"
    exit 0
fi
bash sbatch/start25.sh cifar10 ${PASSTHRU[@]+"${PASSTHRU[@]}"}

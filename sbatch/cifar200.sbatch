#!/bin/bash
# =============================================================================
# CIFAR-10, 200 epochs, standalone. Encode -> reference -> train -> evaluate
# (with t-SNE) -> LaTeX tables, in one self-chaining job that needs no other
# dataset and touches no other run.
#
#     sbatch sbatch/cifar200.sbatch                  # 3h55 links (schedules fast)
#     sbatch --time=24:00:00 sbatch/cifar200.sbatch  # one long link, if allowed
#     bash   sbatch/cifar200.sbatch                  # run here, no chaining
#
# WHY IT DOES NOT COLLIDE WITH AN EXISTING cifar10 RUN
#   Everything is written under the tag ep200, so this run owns
#   work/runs/cifar10__ep200/ and work/results/cifar10__ep200/. The expensive
#   dataset-level artefacts -- work/encoded/cifar10_256 and
#   work/reference/cifar10_32 -- are NOT tagged and are reused as they are,
#   because they depend on the dataset and resolution only. See the comment on
#   tag in scripts/run_pipeline.py. A plain gfm-c10 job writing
#   work/runs/cifar10/ can therefore run at the same time without either one
#   clobbering the other.
#
# THE 3h55 WALL CLOCK
#   Each link submits its successor (--dependency=afterany on itself) BEFORE
#   doing any work, so the chain survives being killed at the wall clock,
#   crashing, or being pre-empted. The successor resumes from train_state.pt:
#   weights, AdamW moments, cosine schedule, epoch and step. When a link finds
#   the work already finished it exits in seconds and the chain ends, costing
#   exactly one no-op job.
#
#   Each link self-submits with the SAME --time it is running under, so
#   sbatch --time=24:00:00 chains 24 h links rather than silently dropping back
#   to the 3h55 in the header below.
#
#   200 epochs is ~156k optimizer steps at batch 64 (781 steps/epoch over
#   CIFAR-10's 50k images), so expect several links unless a long slot is
#   available.
#
# THE RESUME GUARD -- read this before raising GFM_MAX_LINKS
#   Chaining is only safe if train_state.pt is actually being written. If it is
#   not, every link restarts from epoch 0 and the chain burns its whole budget
#   re-training the same first epochs forever. train.py writes that file every
#   --state_every_steps (2000) and at every epoch boundary, so a link that
#   trained for hours and left no state file means resuming is broken. This
#   script checks exactly that and stops the chain with an explanation instead
#   of looping. Do not work around it by raising GFM_MAX_LINKS.
#
# EVALUATION
#   Runs in the same job the moment training reaches EPOCHS, unless too little
#   wall clock is left -- sampling cannot resume, so starting it with ten
#   minutes to go would throw the work away, and the already-queued successor
#   does it instead. NSTEPS defaults to 50 here, NOT to run_pipeline.py's 3:
#   three RK4 steps cannot produce an image, whatever the model has learned.
#   Note NFE = 4 x NSTEPS for rk4; the metrics.json now records the measured
#   count under timing.nfe_per_image and the raw step count under
#   timing.solver_steps, so the two can never be confused in the paper.
#   t-SNE comes from evaluation/evaluate_gfm.py unless --skip_tsne is passed,
#   which this script never does.
#
# OUTPUT
#   work/results/cifar10__ep200/metrics.json    FID, IS, KID, MMD, timings
#   work/results/cifar10__ep200/samples/        10k PNGs at 32x32
#   work/results/cifar10__ep200/tsne_*.png|pdf  joint + by-class t-SNE
#   paper/generated/*.tex                       results, timing, loss, config
#   paper/figures/*                             curves + copied t-SNE/samples
#
# ENVIRONMENT
#   EPOCHS=200          how far to train
#   NSTEPS=25           ODE *steps*. With rk4 that is 100 NFE per image --
#                       the solver calls the velocity field four times per
#                       step (utils/utils.py), so NFE = 4 x NSTEPS. 100 NFE is
#                       the range Lipman et al. (arXiv:2210.02747) report for
#                       CIFAR-10, and costs ~75 min to sample 10k images here.
#                       NSTEPS=50 is 200 NFE and ~2.5 h -- still fits a 3h55
#                       link, but only if the link starts it fresh.
#   EVAL_SAMPLES=10000  images sampled for the metrics
#   TAG=ep200           run/result directory suffix
#   GFM_MAX_LINKS=40    give up after this many links
#   WORK_DIR, BATCH_SIZE, MICRO_BATCH, ... as in sbatch/run_gfm.sbatch
#
# WATCHING / STOPPING
#   tail -f sbatch/logs/c200_*.out
#   squeue shows these as c200; run_all25 cifar10 links are gfm-c10-*.
#   touch $WORK_DIR/.cifar200/stop     # stop chaining after the current link
# =============================================================================
# NOT gfm-*: run_gfm.sbatch renames any job called gfm, run_gfm.sbatch or
# gfm-* to gfm-<dataset>-<phase>, and <dataset> comes from the dataset rather
# than from this chain -- so a gfm-c200 job shows up in squeue as gfm-c10-trn,
# identical to a run_all25 cifar10 link. Then you cannot tell the two chains
# apart and cancelling the right one becomes guesswork. A name outside that
# pattern is kept as-is ("keeping explicit job name").
#SBATCH --job-name=c200
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
#SBATCH --time=03:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_3090|rtx_4090|rtx_6000|rtx_pro_6000

# NOT set -e: a failing stage must still reach the bookkeeping at the bottom,
# which is what decides whether the chain continues and says why.
set -uo pipefail

REPO="${GFM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO" >&2; exit 1; }
if [ ! -f train.py ] || [ ! -f sbatch/run_gfm.sbatch ]; then
    echo "ERROR: $REPO is not the GraphFlowMatching repo (no train.py)." >&2
    echo "       Submit from the repo root, or set GFM_REPO=/path/to/repo." >&2
    exit 1
fi
export GFM_REPO="$REPO"
mkdir -p sbatch/logs

if [ -f sbatch/site.env ]; then
    set -a
    # shellcheck disable=SC1091
    . sbatch/site.env
    set +a
fi
export WORK_DIR="${WORK_DIR:-$PWD/work}"

export EPOCHS="${EPOCHS:-200}"
export NSTEPS="${NSTEPS:-25}"
export EVAL_SAMPLES="${EVAL_SAMPLES:-10000}"
TAG="${TAG:-ep200}"
GFM_MAX_LINKS="${GFM_MAX_LINKS:-40}"

DATASET=cifar10
VARIANT="cifar10__${TAG}"
# MUST stay exactly "CIFAR-10". build_paper.py:copy_figures derives the figure
# filename from this label (lower-case, spaces and dashes to underscores), and
# paper/main.tex includes those figures by name: tsne_cifar_10,
# tsne_cifar_10_byclass, samples_cifar_10. Anything else here -- "CIFAR-10 (200
# ep)", say -- copies the t-SNE to a name main.tex never looks for, and the
# figure silently renders as "(pending)". The epoch count is already a column
# in the loss and timing tables.
LABEL="CIFAR-10"
STATE_DIR="$WORK_DIR/.cifar200"
RUN_DIR="$WORK_DIR/runs/$VARIANT/models"
RES_DIR="$WORK_DIR/results/$VARIANT"
RES_JSON="$RES_DIR/metrics.json"
EPOCH_CSV="$RUN_DIR/train_log_epochs.csv"
STATE_PT="$RUN_DIR/train_state.pt"
mkdir -p "$STATE_DIR"

LINK=$(( $(cat "$STATE_DIR/links" 2>/dev/null || echo 0) + 1 ))
echo "$LINK" > "$STATE_DIR/links"
log() { echo "[c200 link ${LINK}] $*"; }

# --------------------------------------------------------------------------- #
#  How much of this allocation is left, and how long was it asked for?
# --------------------------------------------------------------------------- #
_hms_to_s() {
    local raw="$1" d=0 h m s
    case "$raw" in
        ''|UNLIMITED|INVALID|NOT_SET) echo 999999; return 0 ;;
    esac
    case "$raw" in *-*) d="${raw%%-*}"; raw="${raw#*-}" ;; esac
    case "$raw" in
        *:*:*) h="${raw%%:*}"; m="$(echo "$raw" | cut -d: -f2)"; s="${raw##*:}" ;;
        *:*)   h=0;            m="${raw%%:*}";                   s="${raw##*:}" ;;
        *)     h=0; m=0; s="$raw" ;;
    esac
    case "${d}${h}${m}${s}" in *[!0-9]*) echo 999999; return 0 ;; esac
    echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}
time_left_s() {
    _hms_to_s "$(squeue -h -j "${SLURM_JOB_ID:-0}" -o '%L' 2>/dev/null | tr -d ' ')"
}
# The successor must ask for the same slot length as this link, or a
# sbatch --time=24:00:00 chain would quietly fall back to the 3h55 header.
job_time_limit() {
    squeue -h -j "${SLURM_JOB_ID:-0}" -o '%l' 2>/dev/null | tr -d ' '
}
fmt_left() { printf '%dh%02dm' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )); }

# --------------------------------------------------------------------------- #
#  What is already done? Answered from artefacts, never from bookkeeping.
# --------------------------------------------------------------------------- #
have_latents()   { compgen -G "$WORK_DIR/encoded/cifar10_*/metadata.json" >/dev/null 2>&1; }
have_reference() { compgen -G "$WORK_DIR/reference/cifar10_*_meta/reference_meta.json" >/dev/null 2>&1; }
epochs_done() {
    local last
    [ -s "$EPOCH_CSV" ] || { echo 0; return 0; }
    last="$(tail -n 1 "$EPOCH_CSV" | cut -d, -f1)"
    case "$last" in ''|*[!0-9]*) echo 0 ;; *) echo "$last" ;; esac
}
training_done() { [ "$(epochs_done)" -ge "$EPOCHS" ]; }
eval_done()     { [ -s "$RES_JSON" ]; }
all_done()      { eval_done && training_done; }

run_stage() {
    local phase="$1" rc=0
    shift
    log ">>> ${phase}  ($(fmt_left "$(time_left_s)") left)"
    bash sbatch/run_gfm.sbatch "$DATASET" "$phase" "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then
        log "<<< ${phase} finished"
    else
        log "<<< ${phase} exited ${rc}"
    fi
    return "$rc"
}

echo "=================================================="
echo " CIFAR-10 standalone  link=${LINK}  epochs=${EPOCHS}  nsteps=${NSTEPS}"
echo " job=${SLURM_JOB_ID:-none}  host=$(hostname)"
echo " variant=${VARIANT}"
echo " run_dir=${RUN_DIR}"
echo " res_dir=${RES_DIR}"
echo " slot=$(job_time_limit)  left=$(fmt_left "$(time_left_s)")"
echo " state: latents=$(have_latents && echo yes || echo no)" \
     "reference=$(have_reference && echo yes || echo no)" \
     "epochs=$(epochs_done)/${EPOCHS}" \
     "evaluated=$(eval_done && echo yes || echo no)"
echo "=================================================="

# --------------------------------------------------------------------------- #
#  Submit the successor FIRST, so the chain survives this link being killed.
# --------------------------------------------------------------------------- #
submit_successor() {
    local out id tl
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        log "not under Slurm -- doing this link's work here, no chaining"
        return 0
    fi
    if [ -f "$STATE_DIR/stop" ]; then
        log "stop file present ($STATE_DIR/stop) -- not continuing the chain"
        return 0
    fi
    if [ "$LINK" -ge "$GFM_MAX_LINKS" ]; then
        log "ERROR: reached GFM_MAX_LINKS=${GFM_MAX_LINKS} without finishing."
        log "       Read the last few logs before raising it."
        return 1
    fi
    tl="$(job_time_limit)"
    out="$(sbatch --job-name=c200 \
                  ${tl:+--time="$tl"} \
                  --dependency="afterany:${SLURM_JOB_ID}" \
                  ${GFM_SBATCH_EXTRA:+$GFM_SBATCH_EXTRA} \
                  sbatch/cifar200.sbatch 2>&1)" || {
        log "ERROR: could not submit the next link:"
        printf '%s\n' "$out" >&2
        return 1
    }
    id="$(printf '%s\n' "$out" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n1)"
    log "next link queued as ${id:-?} (--time=${tl:-header}, after ${SLURM_JOB_ID})"
}

if all_done; then
    log "already finished -- no successor needed"
else
    submit_successor || true
fi

# --------------------------------------------------------------------------- #
#  Stages
# --------------------------------------------------------------------------- #
MIN_STAGE_S=2400        # encode / reference: neither resumes
MIN_TRAIN_S=1200        # below this a train link mostly repeats itself
# Sampling cannot resume: a link killed halfway through throws away every
# image it made. At NSTEPS=25 / 100 NFE, 10k images is ~75 min plus ~5 min of
# metrics, so 2 h of headroom means a link either does the whole thing or does
# not start it. Raise this if you raise NSTEPS.
MIN_EVAL_S=7200

if [ ! -f "$STATE_DIR/warmup.ok" ]; then
    if bash sbatch/run_gfm.sbatch cifar10 warmup; then
        : > "$STATE_DIR/warmup.ok"
    fi
fi

if ! have_latents; then
    if [ "$(time_left_s)" -lt "$MIN_STAGE_S" ]; then
        log "skipping encode: only $(fmt_left "$(time_left_s)") left and it cannot resume"
    else
        run_stage encode || true
    fi
fi

if have_latents && ! have_reference; then
    if [ "$(time_left_s)" -lt "$MIN_STAGE_S" ]; then
        log "skipping reference: only $(fmt_left "$(time_left_s)") left"
    else
        run_stage reference || true
    fi
fi

RAN_TRAIN=0
if have_latents && ! training_done; then
    if [ "$(time_left_s)" -lt "$MIN_TRAIN_S" ]; then
        log "skipping train: only $(fmt_left "$(time_left_s)") left"
    else
        RAN_TRAIN=1
        run_stage train --tag "$TAG" || true
        log "after this link: $(epochs_done)/${EPOCHS} epochs"
    fi
fi

# --------------------------------------------------------------------------- #
#  The resume guard.
#
#  train.py writes train_state.pt every --state_every_steps (2000) and at every
#  epoch boundary. A link that ran the train stage, did not finish, and left no
#  state file cannot be resumed -- the next link would start again from epoch 0,
#  and so would every link after it. Stop the chain and say so, rather than
#  spending the whole budget re-learning epoch 1.
# --------------------------------------------------------------------------- #
if [ "$RAN_TRAIN" -eq 1 ] && ! training_done && [ ! -f "$STATE_PT" ]; then
    : > "$STATE_DIR/stop"
    {
        echo "----------------------------------------------------------------"
        echo " FATAL: training ran but wrote no $STATE_PT"
        echo ""
        echo " train.py saves that file every --state_every_steps (2000) and at"
        echo " every epoch boundary, so its absence means resuming is broken."
        echo " Chaining on would restart from epoch 0 in every link and never"
        echo " reach ${EPOCHS} epochs."
        echo ""
        echo " A stop file has been written, so no further link will run:"
        echo "     $STATE_DIR/stop"
        echo ""
        echo " Look for [ckpt] and [resume] lines in this log. Then either fix"
        echo " the cause, or use one slot big enough to need no resume at all:"
        echo "     rm $STATE_DIR/stop"
        echo "     sbatch --time=24:00:00 sbatch/cifar200.sbatch"
        echo "----------------------------------------------------------------"
    } >&2
    exit 1
fi

if training_done && ! eval_done; then
    if [ "$(time_left_s)" -lt "$MIN_EVAL_S" ]; then
        log "training done, but only $(fmt_left "$(time_left_s)") left and"
        log "  sampling cannot resume -- the queued successor evaluates it"
    else
        # Evaluation that keeps not finishing is the one way this chain can
        # spin: sampling cannot resume, so a slot too short to hold it makes
        # every link start over and throw the images away. Count the attempts
        # and stop rather than repeat that forever.
        TRIES=$(( $(cat "$STATE_DIR/eval_tries" 2>/dev/null || echo 0) + 1 ))
        echo "$TRIES" > "$STATE_DIR/eval_tries"
        log "evaluation attempt ${TRIES}"

        # t-SNE is on by default in evaluation/evaluate_gfm.py; --skip_tsne is
        # never passed here. NSTEPS reaches run_pipeline.py through COMMON in
        # run_gfm.sbatch.
        run_stage evaluate --tag "$TAG" || true

        if ! eval_done && [ "$TRIES" -ge 2 ]; then
            : > "$STATE_DIR/stop"
            {
                echo "------------------------------------------------------------"
                echo " FATAL: evaluation has failed to finish ${TRIES} times."
                echo ""
                echo " Sampling ${EVAL_SAMPLES} images at ${NSTEPS} steps"
                echo " (= $(( NSTEPS * 4 )) NFE with rk4) does not fit the slot this"
                echo " chain is using, and it cannot resume, so each link throws"
                echo " away what it sampled. Chaining on would repeat that."
                echo ""
                echo " A stop file has been written: $STATE_DIR/stop"
                echo ""
                echo " Training is finished and safe. Evaluate it in one big slot:"
                echo "     rm $STATE_DIR/stop $STATE_DIR/eval_tries"
                echo "     sbatch --time=12:00:00 sbatch/cifar200.sbatch"
                echo " or cut the cost -- NSTEPS=10 is 40 NFE, EVAL_SAMPLES=5000"
                echo " halves it again:"
                echo "     NSTEPS=10 sbatch --time=03:55:00 sbatch/cifar200.sbatch"
                echo "------------------------------------------------------------"
            } >&2
            exit 1
        fi
    fi
fi

# --------------------------------------------------------------------------- #
#  Report: curves, figures and the LaTeX tables, for this run alone.
# --------------------------------------------------------------------------- #
if all_done; then
    log "building the report for ${VARIANT}"
    if [ -s "$RUN_DIR/train_log_steps.csv" ]; then
        python scripts/plot_training_curves.py \
            --run "${LABEL}=${RUN_DIR}" --out_dir paper/figures || true
    fi
    python scripts/build_paper.py \
        --run "${LABEL}=${RUN_DIR}" --result "${LABEL}=${RES_DIR}" || true
    echo
    log "done. Tables in paper/generated/ (results, timing, loss, config)."
    log "  PDF:  cd paper && latexmk -pdf main.tex"
fi

echo "=================================================="
echo " c200 link ${LINK} done"
echo " epochs=$(epochs_done)/${EPOCHS}  evaluated=$(eval_done && echo yes || echo no)"
echo "=================================================="

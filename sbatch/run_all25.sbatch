#!/bin/bash
# =============================================================================
# ONE submission that carries all three datasets to 25 epochs, evaluates them
# and writes the report -- under a hard 4 h wall-clock cap.
#
#     sbatch sbatch/run_all25.sbatch
#
# That is the whole command. It expands into a job array of three tasks (one
# per dataset, running in parallel), and each task keeps re-submitting itself
# until its dataset is finished:
#
#   task 0  cifar10      encode -> reference -> train (25 ep) -> evaluate --.
#   task 1  imagenet-lt  encode -> reference -> train (25 ep) -> evaluate --+-> report
#   task 2  imagenet     encode -> reference -> train (25 ep) -> evaluate --'
#
# WHY IT SELF-CHAINS INSTEAD OF PRE-SUBMITTING N LINKS
#   sbatch/run25.sh submits a fixed number of links, so it has to GUESS how
#   many 4 h slots 25 epochs will need -- 28 for ImageNet on the slowest card
#   the constraint allows. Guess low and the run stops short; guess high and
#   the queue carries jobs nobody needs. This script does not guess: every link
#   works until Slurm kills it, and the next link picks up from
#   train_state.pt. Training simply runs for as long as it needs.
#
#   Each link submits its successor with --dependency=afterany BEFORE it starts
#   any real work, so the chain survives a link that is killed at the wall
#   clock, crashes, or is pre-empted. When a link finds nothing left to do it
#   exits in seconds without submitting anything, and the chain ends. The cost
#   of that design is exactly one no-op job per dataset at the end.
#
# THE GPU
#   Same --constraint and --gres as sbatch/run_gfm.sbatch, i.e. the four card
#   types this project is set up for. Nothing else needs to be passed.
#
# WATCHING IT
#   squeue -u $USER -o '%.10i %.20j %.2t %.11M %.20E %R'
#   Each link renames itself to gfm-<dataset>-<phase> while a stage runs, so
#   the NAME column says which dataset is in which stage right now.
#   tail -f sbatch/logs/gfm25_*_*.out
#
# WHERE THE STATE LIVES
#   $WORK_DIR/.gfm25/ -- one small text file per dataset: how many links have
#   run, what the last link achieved, and a stamp per finished stage. Delete
#   that directory to make the next submission re-check everything from
#   scratch (it re-checks the artefacts either way; the stamps only make the
#   check cheaper and the log clearer).
#
# STOPPING IT
#   scancel -u $USER --name=gfm25          # and any renamed links:
#   scancel -u $USER --name=gfm-c10-trn    # ... etc
#   Cancelling one link does NOT stop the chain: its successor is already
#   queued and will start the next stage. To stop for good, cancel twice, or
#   `touch $WORK_DIR/.gfm25/stop` -- every link checks for that file and exits
#   without submitting a successor.
#
# ENVIRONMENT (all optional, and all inherited by every later link)
#   EPOCHS=25                  epochs per dataset
#   GFM_DATASETS="cifar10 imagenet-lt imagenet"
#                              which datasets, in array-index order. Submit a
#                              subset with e.g.
#                                GFM_DATASETS=cifar10 sbatch --array=0 \
#                                    sbatch/run_all25.sbatch
#   GFM_SBATCH_EXTRA="--exclude=bad-node-01"
#                              extra sbatch options for every self-submission
#   GFM_MAX_LINKS=60           give up after this many links for one dataset
#   GFM_MAX_STALLS=3           give up after this many links in a row that made
#                              no measurable progress (a crash loop)
#   WORK_DIR, DATA_ROOT_IMAGENET, BATCH_SIZE, MICRO_BATCH, ... as documented in
#   sbatch/run_gfm.sbatch; this script only sets EPOCHS and forwards the rest.
#
# WHAT IT DOES NOT DO
#   encode cannot resume -- it writes latent shards and only then the
#   metadata.json that marks them usable -- so a timed-out encode restarts from
#   the beginning. ImageNet encode is 2-4 h at 8 workers, hence the 16 CPUs
#   below, and it is started only when a link has >40 min of wall clock left.
#   If it still cannot finish inside one slot, run that one stage without a
#   cap: bash sbatch/run_gfm.sbatch imagenet encode
# =============================================================================
#SBATCH --job-name=gfm25
#SBATCH --output=sbatch/logs/%x_%A_%a.out
#SBATCH --error=sbatch/logs/%x_%A_%a.err
#SBATCH --time=03:55:00
#SBATCH --array=0-2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_3090|rtx_4090|rtx_6000|rtx_pro_6000

# NOT set -e: a stage that fails must still reach the bookkeeping at the end of
# this script, which is what decides whether the chain continues and why.
set -uo pipefail

# Slurm may run this script from a spool copy, so $0 is not the repo. The
# submission directory is, and it is carried forward explicitly so that later
# links do not depend on where they were submitted from.
REPO="${GFM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO" >&2; exit 1; }
if [ ! -f train.py ] || [ ! -f sbatch/run_gfm.sbatch ]; then
    echo "ERROR: $REPO is not the GraphFlowMatching repo (no train.py)." >&2
    echo "       Submit from the repo root, or set GFM_REPO=/path/to/repo." >&2
    exit 1
fi
export GFM_REPO="$REPO"
mkdir -p sbatch/logs

export EPOCHS="${EPOCHS:-25}"
GFM_MAX_LINKS="${GFM_MAX_LINKS:-60}"
GFM_MAX_STALLS="${GFM_MAX_STALLS:-3}"
# shellcheck disable=SC2206  # word splitting is the point: these are options
SBATCH_EXTRA=(${GFM_SBATCH_EXTRA:-})

# Which dataset is this task? The list is also what the report waits for.
read -r -a DATASETS <<< "${GFM_DATASETS:-cifar10 imagenet-lt imagenet}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#DATASETS[@]}" ]; then
    echo "array index ${IDX} is beyond GFM_DATASETS (${DATASETS[*]}) -- nothing to do"
    exit 0
fi
DATASET="${DATASETS[$IDX]}"
SLUG="${DATASET//-/_}"

# WORK_DIR is resolved the same way run_gfm.sbatch resolves it, and for the
# same reason: site.env is where a cluster puts its scratch path. Only the
# value is needed here -- everything else in that file is for the stage jobs.
if [ -f sbatch/site.env ]; then
    set -a
    # shellcheck disable=SC1091
    . sbatch/site.env
    set +a
fi
export WORK_DIR="${WORK_DIR:-$PWD/work}"
STATE_DIR="$WORK_DIR/.gfm25"
mkdir -p "$STATE_DIR"

RUN_DIR="$WORK_DIR/runs/$SLUG/models"
RES_JSON="$WORK_DIR/results/$SLUG/metrics.json"
EPOCH_CSV="$RUN_DIR/train_log_epochs.csv"
STEP_CSV="$RUN_DIR/train_log_steps.csv"
STATE_PT="$RUN_DIR/train_state.pt"

log() { echo "[gfm25 ${DATASET} link ${LINK:-?}] $*"; }

# --------------------------------------------------------------------------- #
#  How much of this allocation is left?
#
#  Every decision to start a stage is really "will it get anywhere before
#  Slurm kills me". squeue knows; if it does not answer (no Slurm on the node,
#  an unparsable format) the safe assumption is "plenty", because starting a
#  stage too late merely repeats work the next link redoes anyway.
# --------------------------------------------------------------------------- #
time_left_s() {
    local raw d h m s
    raw="$(squeue -h -j "${SLURM_JOB_ID:-0}" -o '%L' 2>/dev/null | tr -d ' ')"
    case "$raw" in
        ''|UNLIMITED|INVALID|NOT_SET) echo 999999; return 0 ;;
    esac
    d=0
    case "$raw" in *-*) d="${raw%%-*}"; raw="${raw#*-}" ;; esac
    case "$raw" in
        *:*:*) h="${raw%%:*}"; m="$(echo "$raw" | cut -d: -f2)"; s="${raw##*:}" ;;
        *:*)   h=0;            m="${raw%%:*}";                   s="${raw##*:}" ;;
        *)     h=0; m=0; s="$raw" ;;
    esac
    case "${d}${h}${m}${s}" in
        *[!0-9]*) echo 999999; return 0 ;;
    esac
    echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

fmt_left() { printf '%dh%02dm' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )); }

# --------------------------------------------------------------------------- #
#  What is already done?
#
#  Answered from the artefacts, not from bookkeeping: a dataset encoded last
#  month, or trained by sbatch/run25.sh, must not be redone just because this
#  script has no stamp for it.
# --------------------------------------------------------------------------- #
have_latents() {
    compgen -G "$WORK_DIR/encoded/${SLUG}_*/metadata.json" >/dev/null 2>&1
}

have_reference() {
    [ -f "$STATE_DIR/${SLUG}.reference.ok" ] && return 0
    compgen -G "$WORK_DIR/reference/${SLUG}_*_meta/reference_meta.json" >/dev/null 2>&1
}

# Last epoch number in the training log. log_epoch() is called with epoch + 1,
# so this is a count of finished epochs, and it survives the resume because the
# CSV is appended to rather than rewritten.
epochs_done() {
    local last
    [ -s "$EPOCH_CSV" ] || { echo 0; return 0; }
    last="$(tail -n 1 "$EPOCH_CSV" | cut -d, -f1)"
    case "$last" in ''|*[!0-9]*) echo 0 ;; *) echo "$last" ;; esac
}

training_done() { [ "$(epochs_done)" -ge "$EPOCHS" ]; }
eval_done()     { [ -s "$RES_JSON" ]; }
gave_up()       { [ -f "$STATE_DIR/${SLUG}.gaveup" ]; }

dataset_done() { { eval_done && training_done; } || gave_up; }

# --------------------------------------------------------------------------- #
#  Did the previous link achieve anything?
#
#  An ImageNet epoch is ~3.6 h on a 3090, so a 3h55 link can easily finish ZERO
#  epochs and still have done four hours of useful work. Progress is therefore
#  measured on the step counter and the resume file, never on epochs alone --
#  otherwise a healthy long run would look like a crash loop.
# --------------------------------------------------------------------------- #
progress_signature() {
    local step=0 mtime=0 e
    if [ -s "$STEP_CSV" ]; then
        step="$(tail -n 1 "$STEP_CSV" | cut -d, -f1)"
        case "$step" in ''|*[!0-9]*) step=0 ;; esac
    fi
    [ -f "$STATE_PT" ] && mtime="$(date -r "$STATE_PT" +%s 2>/dev/null || echo 0)"
    e="$(epochs_done)"
    echo "e${e}:s${step}:m${mtime}:l$(have_latents && echo 1 || echo 0):r$(have_reference && echo 1 || echo 0):v$(eval_done && echo 1 || echo 0)"
}

# --------------------------------------------------------------------------- #
#  Run one stage of the pipeline.
#
#  run_gfm.sbatch is the single place that knows about venv activation, the
#  ImageNet root, micro-batch sizing and the CUDA preflight, so stages go
#  through it rather than calling run_pipeline.py directly. It also renames the
#  running job to gfm-<dataset>-<phase>, which is what makes squeue readable
#  while a chain is in flight.
# --------------------------------------------------------------------------- #
run_stage() {
    local phase="$1" rc=0
    log ">>> ${phase}  ($(fmt_left "$(time_left_s)") left)"
    bash sbatch/run_gfm.sbatch "$DATASET" "$phase" || rc=$?
    if [ "$rc" -eq 0 ]; then
        : > "$STATE_DIR/${SLUG}.${phase}.ok"
        log "<<< ${phase} finished"
    else
        # Killed at the wall clock (SIGTERM -> 143) is the expected way for a
        # training link to end, and says nothing about whether it went well.
        log "<<< ${phase} exited ${rc}"
    fi
    return "$rc"
}

# --------------------------------------------------------------------------- #
#  Bookkeeping: link number, stall counter
# --------------------------------------------------------------------------- #
LINK=$(( $(cat "$STATE_DIR/${SLUG}.links" 2>/dev/null || echo 0) + 1 ))
echo "$LINK" > "$STATE_DIR/${SLUG}.links"
# The state the PREVIOUS link left behind, and how many links in a row have
# left it untouched. Both are only compared once this link has done its work,
# at the bottom of the script -- comparing them here would compare the previous
# link's end state with itself.
PREV_SIG="$(cat "$STATE_DIR/${SLUG}.sig" 2>/dev/null || echo none)"
STALLS="$(cat "$STATE_DIR/${SLUG}.stalls" 2>/dev/null || echo 0)"

echo "=================================================="
echo " gfm25  dataset=${DATASET}  link=${LINK}  epochs=${EPOCHS}"
echo " host=$(hostname)  job=${SLURM_JOB_ID:-none}  array=${SLURM_ARRAY_JOB_ID:-none}[${IDX}]"
echo " work_dir=${WORK_DIR}"
echo " time limit=$(fmt_left "$(time_left_s)")"
echo " state: latents=$(have_latents && echo yes || echo no)"                 \
     "reference=$(have_reference && echo yes || echo no)"                     \
     "epochs=$(epochs_done)/${EPOCHS}"                                        \
     "evaluated=$(eval_done && echo yes || echo no)"
echo "=================================================="

if [ "$STALLS" -gt 0 ] && ! dataset_done; then
    log "note: ${STALLS} previous link(s) in a row made no progress"
fi

# --------------------------------------------------------------------------- #
#  Submit the successor FIRST, so the chain survives this link being killed.
# --------------------------------------------------------------------------- #
submit_successor() {
    local out id
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        # Run by hand rather than submitted: do the work in this shell and do
        # not spawn jobs. There is nothing to depend on, and a chain launched
        # from a login node would keep running after the terminal is gone.
        log "not running under Slurm -- doing this link's work here, no chaining"
        return 0
    fi
    if [ -f "$STATE_DIR/stop" ]; then
        log "stop file present ($STATE_DIR/stop) -- not continuing the chain"
        return 0
    fi
    if [ "$LINK" -ge "$GFM_MAX_LINKS" ]; then
        log "ERROR: reached GFM_MAX_LINKS=${GFM_MAX_LINKS} without finishing."
        log "       Raise GFM_MAX_LINKS, or look for the real problem in the"
        log "       logs of the last few links before resubmitting."
        : > "$STATE_DIR/${SLUG}.gaveup"
        return 1
    fi
    if [ "$STALLS" -ge "$GFM_MAX_STALLS" ]; then
        log "ERROR: ${STALLS} links in a row made no progress -- this is a crash"
        log "       loop, not slow training. Not submitting another link."
        log "       Read sbatch/logs/gfm25_*_${IDX}.err, fix, then resubmit:"
        log "         rm ${STATE_DIR}/${SLUG}.stalls"
        log "         sbatch --array=${IDX} sbatch/run_all25.sbatch"
        : > "$STATE_DIR/${SLUG}.gaveup"
        return 1
    fi
    out="$(sbatch --array="${IDX}" \
                  --dependency="afterany:${SLURM_JOB_ID}" \
                  ${SBATCH_EXTRA[@]+"${SBATCH_EXTRA[@]}"} \
                  sbatch/run_all25.sbatch 2>&1)" || {
        log "ERROR: could not submit the next link:"
        printf '%s\n' "$out" >&2
        return 1
    }
    id="$(printf '%s\n' "$out" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n1)"
    log "next link queued as ${id:-?} (starts after ${SLURM_JOB_ID})"
}

if dataset_done; then
    log "already finished -- no successor needed"
else
    submit_successor || true
fi

# --------------------------------------------------------------------------- #
#  The stages. Each one is skipped if it is already done, or if too little of
#  the allocation is left for it to get anywhere; the successor picks it up.
# --------------------------------------------------------------------------- #
MIN_STAGE_S=2400        # 40 min: encode / reference / evaluate
MIN_TRAIN_S=1200        # 20 min: below this a train link mostly repeats itself

# Weights are shared by all three tasks, so exactly one of them may download
# them. flock makes the other two wait rather than race on the same HF_HOME.
if [ ! -f "$STATE_DIR/warmup.ok" ]; then
    if command -v flock >/dev/null 2>&1; then
        (
            flock 9
            if [ ! -f "$STATE_DIR/warmup.ok" ]; then
                log "caching pretrained weights (once, shared by all datasets)"
                # A compute node without outbound network cannot do this, and
                # that is not fatal here: it only means the weights must
                # already be cached, and the encode stage says so if they are
                # not. Do not stamp it as done in that case.
                bash sbatch/run_gfm.sbatch cifar10 warmup && : > "$STATE_DIR/warmup.ok"
            fi
        ) 9> "$STATE_DIR/warmup.lock"
    else
        bash sbatch/run_gfm.sbatch cifar10 warmup && : > "$STATE_DIR/warmup.ok"
    fi
fi

if ! have_latents; then
    if [ "$(time_left_s)" -lt "$MIN_STAGE_S" ]; then
        log "skipping encode: only $(fmt_left "$(time_left_s)") left, and encode"
        log "  cannot resume -- the next link starts it with a full slot"
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

if have_latents && ! training_done; then
    if [ "$(time_left_s)" -lt "$MIN_TRAIN_S" ]; then
        log "skipping train: only $(fmt_left "$(time_left_s)") left, which would"
        log "  mostly repeat the current epoch after the resume"
    else
        # Expected to be killed at the wall clock. train.py writes
        # train_state.pt every --state_every_steps and at every epoch
        # boundary, and --resume auto (the default) continues from it.
        run_stage train || true
        log "after this link: $(epochs_done)/${EPOCHS} epochs"
    fi
fi

if training_done && ! eval_done; then
    if [ "$(time_left_s)" -lt "$MIN_STAGE_S" ]; then
        log "training complete; leaving evaluate to the next link"
    else
        run_stage evaluate || true
    fi
fi

# --------------------------------------------------------------------------- #
#  Report -- once, when every dataset has finished (or given up).
#
#  build_paper.py and plot_training_curves.py write one global figure set and
#  one set of tables, so two of them running at once would overwrite each
#  other's rows. flock serialises them; the stamp keeps a finished report from
#  being rebuilt by a link that has nothing else to do.
# --------------------------------------------------------------------------- #
END_SIG="$(progress_signature)"
if dataset_done; then
    # A finished dataset changes nothing by design -- that is what the trailing
    # no-op link is for, and calling it a stall would be alarming nonsense.
    STALLS=0
elif [ "$PREV_SIG" != "none" ] && [ "$END_SIG" = "$PREV_SIG" ]; then
    # Nothing moved: not one step, not one stage, not even a state write. Long
    # training is not this -- an ImageNet link that trains for four hours
    # without completing an epoch still advances the step counter.
    STALLS=$(( STALLS + 1 ))
    log "WARNING: this link changed nothing (${STALLS}/${GFM_MAX_STALLS} in a row)"
else
    STALLS=0
fi
echo "$STALLS" > "$STATE_DIR/${SLUG}.stalls"
echo "$END_SIG" > "$STATE_DIR/${SLUG}.sig"

if dataset_done; then
    : > "$STATE_DIR/done_${SLUG}"
    if gave_up; then
        log "giving up on ${DATASET} at $(epochs_done)/${EPOCHS} epochs -- the"
        log "  report will show it as far as it got, and '--' for the rest"
    else
        log "dataset finished: $(epochs_done)/${EPOCHS} epochs, evaluated"
    fi
fi

all_done=1
for d in "${DATASETS[@]}"; do
    [ -f "$STATE_DIR/done_${d//-/_}" ] || all_done=0
done

if [ "$all_done" -eq 1 ]; then
    report() {
        local stamp="$STATE_DIR/report.ok" newer=0
        # Never built before -> build it, even if not one dataset produced
        # metrics: the tables render '--' for whatever is missing, and a run
        # that ended badly is exactly when the summary is worth having.
        [ -f "$stamp" ] || newer=1
        for d in "${DATASETS[@]}"; do
            local j="$WORK_DIR/results/${d//-/_}/metrics.json"
            [ -s "$j" ] || continue
            if [ ! -f "$stamp" ] || [ "$j" -nt "$stamp" ]; then newer=1; fi
        done
        if [ "$newer" -eq 0 ]; then
            log "report is already up to date"
            return 0
        fi
        log "all datasets finished -- building the report"
        if bash sbatch/run_gfm.sbatch all report; then
            : > "$stamp"
            log "report written: paper/generated/*.tex, paper/figures/*"
            log "  results_table.tex carries FID/IS/MMD/KID plus Train (h) and"
            log "  Infer (min) per dataset; build the PDF with:"
            log "    cd paper && latexmk -pdf main.tex"
        else
            log "ERROR: the report stage failed -- see above"
        fi
    }
    if command -v flock >/dev/null 2>&1; then
        ( flock 9; report ) 9> "$STATE_DIR/report.lock"
    else
        report
    fi
else
    still=()
    for d in "${DATASETS[@]}"; do
        [ -f "$STATE_DIR/done_${d//-/_}" ] || still+=("$d")
    done
    log "still running: ${still[*]} -- the last one to finish builds the report"
fi

echo "=================================================="
echo " gfm25 link ${LINK} done: dataset=${DATASET}"
echo " epochs=$(epochs_done)/${EPOCHS}  evaluated=$(eval_done && echo yes || echo no)"
echo "=================================================="

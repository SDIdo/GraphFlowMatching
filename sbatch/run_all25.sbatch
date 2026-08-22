#!/bin/bash
# =============================================================================
# One self-chaining job that carries ONE dataset to 25 epochs, evaluates it in
# the same job the moment training finishes, and -- once the other datasets
# have finished too -- builds the report.
#
#     bash sbatch/start25.sh                    # all three, one job each
#     sbatch sbatch/run_all25.sbatch cifar10    # or just this one
#
# THREE JOBS, NOT A JOB ARRAY
#   start25.sh submits one ordinary job per dataset, so squeue shows three
#   named lines you can tell apart at a glance:
#
#     JOBID     NAME             ST  TIME
#     4812345   gfm-c10-trn      R   1:12:03
#     4812346   gfm-inlt-trn     R   1:11:58
#     4812347   gfm-in1k-enc     R   0:41:22
#
#   The suffix is the stage running right now: run_gfm.sbatch renames the job
#   as each stage starts (enc / ref / trn / evl / rpt), so the NAME column is a
#   live progress display. Widen it -- it defaults to 8 characters:
#     squeue -u $USER -o '%.10i %.20j %.2t %.11M %.20E %R'
#
# HOW IT RUNS LONGER THAN THE CAP
#   Every link asks for --time=03:55:00 and submits its own successor
#   (--dependency=afterany on itself) BEFORE it starts any real work, so the
#   chain survives a link that is killed at the wall clock, crashes or is
#   pre-empted. The successor resumes from <run_dir>/train_state.pt: model,
#   AdamW moments, cosine schedule, epoch and step. Nothing has to guess how
#   many links 25 epochs will take -- when a link finds the work finished it
#   exits in seconds without submitting anything, and the chain ends. The cost
#   is exactly one no-op job per dataset at the end.
#
# EVALUATION
#   Not a separate submission: the link whose training reaches 25 epochs goes
#   straight on to evaluate, in the same job. Only if that link is nearly out
#   of wall clock does it leave evaluation to its successor, which is already
#   queued -- sampling 10k images cannot resume, so starting it with ten
#   minutes left would throw the work away.
#
# THE REPORT
#   The last dataset to finish runs `all report`, which scans WORK_DIR and
#   rebuilds every figure and table from whatever is there, including the
#   Train (h) / Infer (min) columns. GFM_DATASETS says which datasets to wait
#   for; start25.sh sets it to the ones it launched.
#
# WATCHING / STOPPING
#   tail -f sbatch/logs/gfm-c10_*.out
#   touch $WORK_DIR/.gfm25/stop     # every link checks this and stops chaining
#   scancel <jobid>                 # cancels ONE link; its successor still runs
#                                   # (cancel that too, or use the stop file)
#
# STATE
#   $WORK_DIR/.gfm25/ holds one small text file per dataset: link count, the
#   signature of what the last link achieved, a stamp per finished stage.
#   Deleting it changes nothing about what gets re-run -- the stages are
#   decided from the artefacts themselves -- it only resets the counters.
#
# ENVIRONMENT (optional; inherited by every later link)
#   EPOCHS=25                  epochs per dataset
#   GFM_DATASETS="cifar10 imagenet-lt imagenet"
#                              which datasets the report waits for
#   GFM_SBATCH_EXTRA="--exclude=bad-node-01"
#                              extra sbatch options for every self-submission
#   GFM_MAX_LINKS=60           give up after this many links for one dataset
#   GFM_MAX_STALLS=3           give up after this many links in a row that
#                              achieved nothing (a crash loop)
#   WORK_DIR, DATA_ROOT_IMAGENET, BATCH_SIZE, MICRO_BATCH, ... as documented in
#   sbatch/run_gfm.sbatch; this script only sets EPOCHS and forwards the rest.
#
# THE ONE STAGE THAT CANNOT BE CHAINED
#   encode has no resume -- it writes latent shards and only then the
#   metadata.json that marks them usable -- so a timed-out encode starts over.
#   It is therefore only started when a link has more than 40 minutes left.
#   ImageNet encode is 2-4 h at 8 workers; if it cannot fit one slot, run that
#   single stage once without a cap and everything after it chains normally:
#     sbatch --time=08:00:00 --cpus-per-task=32 sbatch/run_gfm.sbatch imagenet encode
# =============================================================================
#SBATCH --job-name=gfm25
#SBATCH --output=sbatch/logs/%x_%j.out
#SBATCH --error=sbatch/logs/%x_%j.err
#SBATCH --time=03:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_3090|rtx_4090|rtx_6000|rtx_pro_6000

# Deliberately the same cpus / mem / partition / constraint as run_gfm.sbatch:
# those are the values already known to get scheduled on this cluster. Asking
# for more cores than a stage needs only makes the job wait longer for a slot.

# NOT set -e: a stage that fails must still reach the bookkeeping at the end of
# this script, which is what decides whether the chain continues and why.
set -uo pipefail

# Slurm may run this script from a spool copy, so $0 is not the repo. The
# submission directory is, and it is carried forward explicitly so later links
# do not depend on where they were submitted from.
REPO="${GFM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO" >&2; exit 1; }
if [ ! -f train.py ] || [ ! -f sbatch/run_gfm.sbatch ]; then
    echo "ERROR: $REPO is not the GraphFlowMatching repo (no train.py)." >&2
    echo "       Submit from the repo root, or set GFM_REPO=/path/to/repo." >&2
    exit 1
fi
export GFM_REPO="$REPO"
mkdir -p sbatch/logs

DATASET="${1:-${GFM_DATASET:-cifar10}}"
case "$DATASET" in
    cifar10|imagenet|imagenet-lt) ;;
    *) echo "ERROR: unknown dataset '$DATASET' (cifar10|imagenet|imagenet-lt)" >&2
       echo "usage: sbatch sbatch/run_all25.sbatch <dataset>" >&2
       exit 2 ;;
esac
export GFM_DATASET="$DATASET"
SLUG="${DATASET//-/_}"

case "$DATASET" in
    cifar10)     SHORT=c10 ;;
    imagenet)    SHORT=in1k ;;
    imagenet-lt) SHORT=inlt ;;
esac
JOB_NAME="gfm-${SHORT}"

export EPOCHS="${EPOCHS:-25}"
GFM_MAX_LINKS="${GFM_MAX_LINKS:-60}"
GFM_MAX_STALLS="${GFM_MAX_STALLS:-3}"
# shellcheck disable=SC2206  # word splitting is the point: these are options
SBATCH_EXTRA=(${GFM_SBATCH_EXTRA:-})
read -r -a REPORT_DATASETS <<< "${GFM_DATASETS:-cifar10 imagenet-lt imagenet}"

# Name the job after its dataset straight away, so squeue is readable while the
# link is still setting up. run_gfm.sbatch narrows it to gfm-<short>-<phase> as
# each stage starts, which is where the live progress display comes from; that
# rename only fires for names matching gfm-*, which this one does by
# construction.
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    scontrol update "JobId=$SLURM_JOB_ID" "Name=$JOB_NAME" 2>/dev/null || true
fi

# WORK_DIR is resolved the way run_gfm.sbatch resolves it, and for the same
# reason: site.env is where a cluster puts its scratch path. Only that value is
# needed here; everything else in the file is for the stage jobs.
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
#  Every decision to start a stage is really "will it get anywhere before Slurm
#  kills me". squeue knows; if it does not answer (no Slurm on the node, an
#  unparsable format) the safe assumption is "plenty", because starting a stage
#  too late merely repeats work the next link redoes anyway.
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
#  month, or trained by an earlier attempt, must not be redone just because
#  this script has no stamp for it.
# --------------------------------------------------------------------------- #
have_latents() {
    compgen -G "$WORK_DIR/encoded/${SLUG}_*/metadata.json" >/dev/null 2>&1
}

have_reference() {
    [ -f "$STATE_DIR/${SLUG}.reference.ok" ] && return 0
    compgen -G "$WORK_DIR/reference/${SLUG}_*_meta/reference_meta.json" >/dev/null 2>&1
}

# Last epoch number in the training log. log_epoch() is called with epoch + 1,
# so this counts finished epochs, and it survives a resume because the CSV is
# appended to rather than rewritten.
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
#  Did this link achieve anything?
#
#  An ImageNet epoch is ~3.6 h on a 3090, so a 3h55 link can finish ZERO epochs
#  and still have done four hours of useful work. Progress is therefore
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
#  Run one stage.
#
#  run_gfm.sbatch is the single place that knows about venv activation, the
#  ImageNet root, micro-batch sizing and the CUDA preflight, so stages go
#  through it rather than calling run_pipeline.py directly. It also renames the
#  running job to gfm-<short>-<phase>, which is what makes squeue readable
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
#  Bookkeeping
# --------------------------------------------------------------------------- #
LINK=$(( $(cat "$STATE_DIR/${SLUG}.links" 2>/dev/null || echo 0) + 1 ))
echo "$LINK" > "$STATE_DIR/${SLUG}.links"
# What the PREVIOUS link left behind, and how many links in a row have left it
# untouched. Compared only at the bottom of this script, once this link has
# done its work: comparing here would compare the previous link's end state
# with itself.
PREV_SIG="$(cat "$STATE_DIR/${SLUG}.sig" 2>/dev/null || echo none)"
STALLS="$(cat "$STATE_DIR/${SLUG}.stalls" 2>/dev/null || echo 0)"

echo "=================================================="
echo " gfm25  dataset=${DATASET}  link=${LINK}  epochs=${EPOCHS}"
echo " job=${SLURM_JOB_ID:-none} (${JOB_NAME})  host=$(hostname)"
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
        log "       Read sbatch/logs/${JOB_NAME}_*.err, fix, then restart with:"
        log "         rm ${STATE_DIR}/${SLUG}.stalls ${STATE_DIR}/${SLUG}.gaveup"
        log "         sbatch sbatch/run_all25.sbatch ${DATASET}"
        : > "$STATE_DIR/${SLUG}.gaveup"
        return 1
    fi
    out="$(sbatch --job-name="$JOB_NAME" \
                  --dependency="afterany:${SLURM_JOB_ID}" \
                  ${SBATCH_EXTRA[@]+"${SBATCH_EXTRA[@]}"} \
                  sbatch/run_all25.sbatch "$DATASET" 2>&1)" || {
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
#  The stages. Each is skipped when it is already done, or when too little of
#  the allocation is left for it to get anywhere; the successor picks it up.
# --------------------------------------------------------------------------- #
MIN_STAGE_S=2400        # 40 min: encode / reference, neither of which resumes
MIN_TRAIN_S=1200        # 20 min: below this a train link mostly repeats itself
MIN_EVAL_S=1800         # 30 min: sampling cannot resume either

# The weights are shared by all three datasets, so the download must happen
# once. flock makes concurrent dataset jobs wait rather than race on the same
# HF_HOME.
if [ ! -f "$STATE_DIR/warmup.ok" ]; then
    warmup() {
        [ -f "$STATE_DIR/warmup.ok" ] && return 0
        log "caching pretrained weights (once, shared by all datasets)"
        # A compute node with no outbound network cannot do this, and that is
        # not fatal here: it only means the weights must already be cached, and
        # encode says so loudly if they are not. Do not stamp it done then.
        bash sbatch/run_gfm.sbatch cifar10 warmup && : > "$STATE_DIR/warmup.ok"
    }
    if command -v flock >/dev/null 2>&1; then
        ( flock 9; warmup ) 9> "$STATE_DIR/warmup.lock"
    else
        warmup
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
        # train_state.pt every --state_every_steps and at every epoch boundary,
        # and --resume auto (the default) continues from it.
        run_stage train || true
        if training_done; then
            log "training complete: $(epochs_done)/${EPOCHS} epochs"
        else
            log "after this link: $(epochs_done)/${EPOCHS} epochs"
        fi
    fi
fi

# Straight on to evaluation in this same job, which is the usual case: training
# has just finished and there is wall clock left.
if training_done && ! eval_done; then
    if [ "$(time_left_s)" -lt "$MIN_EVAL_S" ]; then
        log "training is complete, but only $(fmt_left "$(time_left_s)") is left"
        log "  and sampling cannot resume -- the next link (already queued)"
        log "  evaluates with a full slot"
    else
        run_stage evaluate || true
    fi
fi

# --------------------------------------------------------------------------- #
#  Progress accounting for the next link
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

# --------------------------------------------------------------------------- #
#  Report -- once, when every dataset has finished (or given up).
#
#  build_paper.py and plot_training_curves.py write one global figure set and
#  one set of tables, so two of them at once would overwrite each other's rows.
#  flock serialises the three dataset jobs; the stamp keeps a finished report
#  from being rebuilt by a link that has nothing else to do.
# --------------------------------------------------------------------------- #
all_done=1
for d in "${REPORT_DATASETS[@]}"; do
    [ -f "$STATE_DIR/done_${d//-/_}" ] || all_done=0
done

if [ "$all_done" -eq 1 ]; then
    report() {
        local stamp="$STATE_DIR/report.ok" newer=0 d j
        # Never built before -> build it, even if not one dataset produced
        # metrics: the tables render '--' for whatever is missing, and a run
        # that ended badly is exactly when the summary is worth having.
        [ -f "$stamp" ] || newer=1
        for d in "${REPORT_DATASETS[@]}"; do
            j="$WORK_DIR/results/${d//-/_}/metrics.json"
            [ -s "$j" ] || continue
            if [ ! -f "$stamp" ] || [ "$j" -nt "$stamp" ]; then newer=1; fi
        done
        if [ "$newer" -eq 0 ]; then
            log "report is already up to date"
            return 0
        fi
        log "every dataset has finished -- building the report"
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
    for d in "${REPORT_DATASETS[@]}"; do
        [ -f "$STATE_DIR/done_${d//-/_}" ] || still+=("$d")
    done
    log "still running: ${still[*]} -- the last one to finish builds the report"
fi

echo "=================================================="
echo " gfm25 link ${LINK} done: dataset=${DATASET}"
echo " epochs=$(epochs_done)/${EPOCHS}  evaluated=$(eval_done && echo yes || echo no)"
echo "=================================================="

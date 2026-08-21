#!/bin/bash
# =============================================================================
# Submit run_gfm.sbatch with a validated node-exclusion list.
#
# WHY THIS EXISTS
#   Slurm validates --exclude at submission time and rejects the whole job if a
#   single name is unknown to the controller:
#
#       sbatch: error: Batch job submission failed: Invalid node name specified
#
#   Node lists go stale (nodes get retired, renamed, moved between clusters), so
#   a hard-coded #SBATCH --exclude line is a submission-time landmine. This
#   wrapper expands the wanted list, intersects it with the nodes that actually
#   exist, and passes only the survivors on the sbatch command line (which
#   overrides the directive inside the script).
#
# USAGE
#   sbatch/submit.sh <dataset> [phase] [extra run_pipeline args...]
#   sbatch/submit.sh --dry-run cifar10 warmup      # print, do not submit
#   sbatch/submit.sh --check                       # just audit the lists
#
#   Any sbatch option can be passed through before the dataset:
#   sbatch/submit.sh --time=36:00:00 cifar10
#
# The exclusion is belt-and-braces anyway: --constraint already restricts the
# job to the four wanted GPU types. If the exclusion cannot be validated the
# script still submits, relying on the constraint alone.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."     # repo root
SCRIPT="sbatch/run_gfm.sbatch"

# Nodes we would like to avoid. Edit here, not in run_gfm.sbatch.
WANT_EXCLUDE="cs-1080-[01-05],cs-2080-[01-05],cs-cpu-[05-11],cs-cpu256-01,cs-pheno-[01-12],dt-2080-[01-19],ee-l40s-[01-02],ise-1080-01,ise-2080-[01-03],ise-cpu-intl-[01-28],ise-cpu128-[01-09],ise-cpu128-[11-14],ise-cpu256-[01-32],ise-pheno-[01-12]"

# GPU features the job asks for (must match run_gfm.sbatch).
WANT_FEATURES="rtx_3090 rtx_4090 rtx_6000 rtx_pro_6000"

DRY_RUN=0
CHECK_ONLY=0
SBATCH_OPTS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --check)   CHECK_ONLY=1; shift ;;
        # sbatch options whose value can be a SEPARATE argument. Without this
        # the value is left in place and read as the dataset, so
        # 'submit.sh -J mine cifar10' would submit dataset 'mine'.
        # The --opt=value form needs nothing special; it matches -* below.
        -J|-t|-p|-A|-q|-w|-x|-C|-G|-c|-N|-n|-o|-e|-d|-M|        --job-name|--time|--partition|--account|--qos|--nodelist|--exclude|        --constraint|--gpus|--cpus-per-task|--nodes|--ntasks|--output|--error|        --dependency|--clusters|--mem|--gres|--reservation)
            if [ $# -ge 2 ]; then
                SBATCH_OPTS+=("$1" "$2"); shift 2
            else
                echo "ERROR: $1 needs a value" >&2; exit 2
            fi ;;
        -*)        SBATCH_OPTS+=("$1"); shift ;;
        *)         break ;;
    esac
done

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found; are you on a submit host?" >&2; exit 1; }

# Slurm opens --output/--error BEFORE the job script runs. If this directory
# is missing the job is launched and then dies immediately with no log at
# all, which looks exactly like "it finished but produced nothing".
mkdir -p sbatch/logs

# ---- 1. which of the wanted exclusions actually exist? ----------------------
KNOWN_NODES="$(sinfo -h -N -o '%N' 2>/dev/null | tr -d '\r' | LC_ALL=C sort -u || true)"
if [ -z "$KNOWN_NODES" ]; then
    echo "WARN: could not list nodes via sinfo; submitting without --exclude." >&2
    EXCLUDE=""
else
    WANTED="$(scontrol show hostnames "$WANT_EXCLUDE" 2>/dev/null | tr -d '\r' | LC_ALL=C sort -u || true)"
    if [ -z "$WANTED" ]; then
        echo "WARN: could not expand the exclude list; submitting without it." >&2
        EXCLUDE=""
    else
        # LC_ALL=C everywhere: comm compares bytes, but sort under a UTF-8
        # locale collates punctuation loosely. Mixing the two makes the
        # intersection silently wrong -- usually empty, which quietly discards
        # the whole exclusion.
        VALID="$(LC_ALL=C comm -12 <(echo "$WANTED") <(echo "$KNOWN_NODES") || true)"
        BOGUS="$(LC_ALL=C comm -23 <(echo "$WANTED") <(echo "$KNOWN_NODES") || true)"

        n_want=$(echo "$WANTED" | grep -c . || true)
        n_valid=$(echo "$VALID" | grep -c . || true)
        n_bogus=$(echo "$BOGUS" | grep -c . || true)
        echo "exclude list: ${n_want} requested, ${n_valid} exist, ${n_bogus} unknown"
        if [ "$n_bogus" -gt 0 ]; then
            echo "  dropping these (not in this cluster -- the cause of"
            echo "  'Invalid node name specified'):"
            echo "$BOGUS" | paste -sd, - | fold -s -w 96 | sed 's/^/    /'
        fi
        if [ "$n_valid" -gt 0 ]; then
            # Slurm accepts a plain comma-separated list; no need to re-compress
            # (and `scontrol show hostlist` does not portably read stdin).
            EXCLUDE="$(echo "$VALID" | paste -sd, -)"
        else
            EXCLUDE=""
            echo
            echo "  none of the requested nodes exist here; relying on --constraint alone"
        fi
    fi
fi

# ---- 2. do the requested GPU features exist? -------------------------------
ALL_FEATURES="$(sinfo -h -o '%f' 2>/dev/null | tr ',' '\n' | sed 's/^ *//;s/ *$//' | sort -u || true)"
if [ -n "$ALL_FEATURES" ]; then
    for f in $WANT_FEATURES; do
        if echo "$ALL_FEATURES" | grep -qx "$f"; then
            avail=$(sinfo -h -o '%n %f' 2>/dev/null | grep -c "\b${f}\b" || true)
            echo "  feature ${f}: OK (${avail} node(s))"
        else
            echo "  feature ${f}: NOT DEFINED on this cluster" >&2
        fi
    done
    echo "  (if a feature is missing, fix --constraint in ${SCRIPT};"
    echo "   'sinfo -o \"%20N %10c %10m %25f %10G\"' lists what is available)"
fi

[ "$CHECK_ONLY" -eq 1 ] && exit 0

# ---- 2b. a job name squeue can tell apart ----------------------------------
# Set at submission so it is already correct while the job is PENDING, and so
# Slurm expands %x in the log filenames to it. The job script derives the same
# name at runtime for plain 'sbatch' submissions; an explicit -J here wins over
# both.
HAVE_NAME=0
for opt in ${SBATCH_OPTS[@]+"${SBATCH_OPTS[@]}"}; do
    case "$opt" in -J|-J*|--job-name|--job-name=*) HAVE_NAME=1 ;; esac
done
if [ "$HAVE_NAME" -eq 0 ] && [ $# -ge 1 ]; then
    case "$1" in
        cifar10)     ds=c10 ;;
        imagenet)    ds=in1k ;;
        imagenet-lt) ds=inlt ;;
        *)           ds="$1" ;;
    esac
    case "${2:-all}" in
        encode) ph=enc ;; reference) ph=ref ;; train) ph=trn ;;
        evaluate) ph=evl ;; warmup) ph=wrm ;; report) ph=rpt ;;
        all) ph=all ;; sweep) ph=swp ;;
        *) ph="$(echo "${2:-all}" | tr ',' '\n' | cut -c1 | tr -d '\n')" ;;
    esac
    SBATCH_OPTS+=("--job-name=gfm-${ds}-${ph}")
    echo "job name: gfm-${ds}-${ph}  (squeue -u \$USER -o '%.10i %.20j %.2t %.11M %.6D %R')"
fi

# ---- 3. submit -------------------------------------------------------------
CMD=(sbatch)
[ ${#SBATCH_OPTS[@]} -gt 0 ] && CMD+=("${SBATCH_OPTS[@]}")
[ -n "$EXCLUDE" ] && CMD+=("--exclude=$EXCLUDE")
CMD+=("$SCRIPT" "$@")

echo
if [ -n "$EXCLUDE" ]; then
    echo "submitting: sbatch ${SBATCH_OPTS[*]:-} --exclude=<${n_valid:-0} nodes> $SCRIPT $*"
else
    echo "submitting: sbatch ${SBATCH_OPTS[*]:-} $SCRIPT $*"
fi
if [ "$DRY_RUN" -eq 1 ]; then
    echo "(dry run -- not submitted)"
    exit 0
fi
"${CMD[@]}"

#!/bin/bash
# =============================================================================
# Run a training job that is longer than the cluster's wall-clock limit.
#
# WHY THIS EXISTS
#   A 200-epoch ImageNet-LT run needs ~67 h. If the partition or QOS caps a job
#   at 4 h, no single submission can finish it, and Slurm kills the job the
#   instant it hits the limit. This submits a chain of dependent jobs instead:
#
#       link 1 ---afterany---> link 2 ---afterany---> link 3 ---> ...
#
#   Every link runs the same command, and train.py resumes from
#   <run_dir>/train_state.pt automatically (--resume auto), restoring the model,
#   the AdamW moments, the cosine LR schedule, the epoch and the step counter.
#   So the chain behaves like one long run, not like N independent restarts --
#   which is what --retrain_flow_network would give you (weights only).
#
#   'afterany', not 'afterok': a link that TIMEOUTs exits non-zero, and that is
#   the normal case here -- the next link still has to start.
#
# USAGE
#   sbatch/chain.sh <links> [sbatch options] <dataset> [phase] [extra args]
#
#   # 17 links of 4 h = 68 h of training, enough for 200 epochs of ImageNet-LT
#   bash sbatch/chain.sh 17 --time=04:00:00 imagenet-lt train
#
#   # then evaluate once, after the chain has finished
#   sbatch --time=04:00:00 --dependency=afterany:<last-id> \
#       sbatch/run_gfm.sbatch imagenet-lt evaluate
#
#   Chain the 'train' phase only. encode/reference are one-off and already done
#   by then, and evaluate belongs at the end, not after every link.
#
#   Arguments are forwarded to sbatch/submit.sh, so node exclusion and the
#   gfm-<dataset>-<phase> job naming work exactly as they do there.
#
# BEFORE YOU CHAIN, check what the cap actually is -- one long job is always
# better than a chain:
#   sinfo -o '%20P %10l %10L'                  # partition MaxTime / DefaultTime
#   sacctmgr show qos format=Name,MaxWall      # QOS caps
#   sacct -j <jobid> -X -o JobID,Timelimit,Elapsed,State
# If MaxTime is larger than what you asked for, raise --time instead.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then
    sed -n '2,44p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

LINKS="$1"; shift
case "$LINKS" in
    ''|*[!0-9]*) echo "ERROR: <links> must be a positive integer (got '$LINKS')" >&2; exit 2 ;;
esac
if [ "$LINKS" -lt 1 ] || [ "$LINKS" -gt 100 ]; then
    echo "ERROR: <links> out of range (1..100): $LINKS" >&2
    exit 2
fi

echo "chaining ${LINKS} dependent job(s): $*"
echo

PREV=""
for i in $(seq 1 "$LINKS"); do
    # stdout and stderr both captured: on success only the link line is worth
    # printing, on failure the whole thing is what explains it.
    OK=0
    if [ -z "$PREV" ]; then
        OUT="$(bash sbatch/submit.sh "$@" 2>&1)" || OK=$?
    else
        OUT="$(bash sbatch/submit.sh "--dependency=afterany:$PREV" "$@" 2>&1)" || OK=$?
    fi
    # The job id is the trailing number of sbatch's "Submitted batch job 12345".
    ID="$(printf '%s\n' "$OUT" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n1)"
    if [ "$OK" -ne 0 ] || [ -z "$ID" ]; then
        echo "$OUT" >&2
        echo >&2
        echo "ERROR: link ${i} was not submitted (submit.sh exited ${OK})." >&2
        if [ "$i" -gt 1 ]; then
            echo "       Links 1..$((i - 1)) ARE queued and will still run." >&2
            echo "       Fix the problem and chain the rest onto ${PREV}, or" >&2
            echo "       scancel them." >&2
        fi
        exit 1
    fi
    if [ -z "$PREV" ]; then
        echo "  link ${i}: ${ID}"
    else
        echo "  link ${i}: ${ID}  (starts after ${PREV})"
    fi
    PREV="$ID"
done

echo
echo "last link: ${PREV}"
echo "evaluate when the chain is done:"
echo "  sbatch --dependency=afterany:${PREV} sbatch/run_gfm.sbatch <dataset> evaluate"
echo "watch it with:"
echo "  squeue -u \$USER -o '%.10i %.20j %.2t %.11M %.20E %R'"

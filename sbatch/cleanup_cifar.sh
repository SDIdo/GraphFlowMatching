#!/bin/bash
# =============================================================================
# Cancel every CIFAR-10 job and stop every CIFAR-10 chain, leaving the ImageNet
# and ImageNet-LT chains running.
#
#     bash sbatch/cleanup_cifar.sh            # show what it would do
#     bash sbatch/cleanup_cifar.sh --yes      # actually do it
#
# WHY THIS IS NOT JUST scancel
#   Every link submits its successor BEFORE it starts working, so cancelling a
#   running link leaves the successor queued and the chain carries on. The stop
#   flags have to go down FIRST, then the jobs get cancelled. Do it the other
#   way round and the chain simply regrows.
#
# WHY NOT touch $WORK_DIR/.gfm25/stop
#   That flag is global to run_all25.sbatch: it would stop the ImageNet and
#   ImageNet-LT chains too. run_all25 decides whether to submit a successor
#   with dataset_done(), which is
#       { eval_done && training_done; } || gave_up
#   and gave_up() is just a per-dataset flag file. Writing cifar10.gaveup
#   therefore stops exactly the cifar10 chains and nothing else. It does not
#   stop a link that is ALREADY running -- that is what the cancels are for.
#
# WHICH JOBS COUNT AS CIFAR
#   Not the name. run_gfm.sbatch renames any job matching gfm-* to
#   gfm-<dataset>-<phase>, so a cifar200.sbatch link shows up as gfm-c10-trn,
#   identical to a run_all25 cifar10 link. The submitted script is the reliable
#   discriminator, and scontrol knows it:
#       Command=.../run_all25.sbatch cifar10   -> old chain, runs/cifar10
#       Command=.../cifar200.sbatch            -> new chain, runs/cifar10__ep200
#   Anything whose Command mentions imagenet is left strictly alone.
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

APPLY=0
for a in "$@"; do
    case "$a" in
        --yes|-y) APPLY=1 ;;
        -h|--help) sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done

if [ -f sbatch/site.env ]; then
    set -a
    # shellcheck disable=SC1091
    . sbatch/site.env
    set +a
fi
WORK_DIR="${WORK_DIR:-$PWD/work}"

if ! command -v squeue >/dev/null 2>&1; then
    echo "ERROR: no squeue on this host -- run this on the login node." >&2
    exit 1
fi

echo "=================================================="
echo " CIFAR-10 cleanup"
echo " work_dir : ${WORK_DIR}"
[ "$APPLY" -eq 1 ] && echo " MODE     : APPLY" || echo " MODE     : dry run (pass --yes to apply)"
echo "=================================================="
echo

# --------------------------------------------------------------------------- #
#  1. Classify every job by the script it was submitted from.
# --------------------------------------------------------------------------- #
echo "1. classifying your jobs"
printf '   %-10s %-16s %-4s %s\n' JOBID NAME ST WHAT
CIFAR_IDS=()
KEEP_IDS=()
UNKNOWN_IDS=()
while read -r id; do
    [ -n "$id" ] || continue
    name="$(squeue -h -j "$id" -o '%j' 2>/dev/null | tr -d ' ')"
    state="$(squeue -h -j "$id" -o '%t' 2>/dev/null | tr -d ' ')"
    cmd="$(scontrol show job "$id" 2>/dev/null | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)"
    # scontrol prints Command= as the script plus its arguments, so the dataset
    # argument of run_all25.sbatch is visible here too.
    full="$(scontrol show job "$id" 2>/dev/null | sed -n 's/^ *Command=//p' | head -1)"
    what="?"
    case "$full" in
        *cifar200.sbatch*)          what="cifar200 chain"; CIFAR_IDS+=("$id") ;;
        *run_all25.sbatch*cifar10*) what="run_all25 cifar10"; CIFAR_IDS+=("$id") ;;
        *imagenet-lt*)              what="imagenet-lt (keep)"; KEEP_IDS+=("$id") ;;
        *imagenet*)                 what="imagenet (keep)"; KEEP_IDS+=("$id") ;;
        *run_gfm.sbatch*cifar10*)   what="run_gfm cifar10"; CIFAR_IDS+=("$id") ;;
        *)                          what="UNKNOWN -- left alone"; UNKNOWN_IDS+=("$id") ;;
    esac
    printf '   %-10s %-16s %-4s %s\n' "$id" "${name:0:16}" "$state" "$what"
    [ -n "$cmd" ] || true
done < <(squeue -u "$USER" -h -o '%i')

echo
echo "   cifar to cancel : ${#CIFAR_IDS[@]}"
echo "   keeping         : ${#KEEP_IDS[@]} imagenet"
echo "   unknown         : ${#UNKNOWN_IDS[@]} (never touched -- check these by hand:"
echo "                     scontrol show job <id> | head -20 )"

if [ "${#CIFAR_IDS[@]}" -eq 0 ]; then
    echo
    echo "nothing to do."
    exit 0
fi

# --------------------------------------------------------------------------- #
#  2. Stop flags FIRST, so a link cancelled below cannot leave a successor.
# --------------------------------------------------------------------------- #
echo
echo "2. stop flags (before any cancel, or the chains regrow)"
G="$WORK_DIR/.gfm25/cifar10.gaveup"
C="$WORK_DIR/.cifar200/stop"
if [ "$APPLY" -eq 1 ]; then
    mkdir -p "$WORK_DIR/.gfm25" "$WORK_DIR/.cifar200"
    : > "$G"; echo "   wrote $G"
    : > "$C"; echo "   wrote $C"
else
    echo "   would write $G   (stops ONLY cifar10 run_all25 chains)"
    echo "   would write $C   (stops cifar200 chains)"
fi

# --------------------------------------------------------------------------- #
#  3. Cancel.
# --------------------------------------------------------------------------- #
echo
echo "3. cancelling ${#CIFAR_IDS[@]} CIFAR job(s)"
if [ "$APPLY" -eq 1 ]; then
    scancel "${CIFAR_IDS[@]}" && echo "   scancel ${CIFAR_IDS[*]}"
    sleep 3
else
    echo "   would run: scancel ${CIFAR_IDS[*]}"
fi

# --------------------------------------------------------------------------- #
#  4. What is left.
# --------------------------------------------------------------------------- #
echo
echo "4. queue now"
squeue -u "$USER" -o '%.10i %.16j %.2t %.11M %.20R' 2>/dev/null | sed 's/^/   /'

echo
if [ "$APPLY" -eq 1 ]; then
    cat <<EOF
=================================================="
Done. Before starting a clean run, clear the flags this script set --
otherwise the new chain refuses to spawn its own successors:

    rm -f "$G" "$C" "$WORK_DIR/.cifar200/links" "$WORK_DIR/.cifar200/eval_tries"
    sbatch sbatch/cifar200.sbatch

It will show in squeue as c200 and stay that way.
EOF
else
    echo "Dry run only. Re-run with --yes to apply."
fi

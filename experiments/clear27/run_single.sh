#!/usr/bin/env bash
# clear27: ONE method on one domain, with the oracle as yardstick (run_single.py), from the repository root.
# The progress bar and every stopped step are shown, and also appended to the log.
#
#   bash experiments/clear27/run_single.sh drone --list                       # the method names and sizes
#   bash experiments/clear27/run_single.sh drone "PCMCI+ -> IMDP (IA)"        # the paper's whole grid
#   bash experiments/clear27/run_single.sh drone "PCMCI+ -> IMDP (IA)" --size 5-1 --n 1000 5000 --seeds 0 1
#   ... --save all|largest|none   --workers N|auto   --resume                 # see run_single.py
#   ... --core-share 50   (or CORE_SHARE=50 ...)                              # at most 50% of the physical cores
#   SHOW=0 ...                                                                # log only
# Results: results/clear27_single/<domain>/<method>_<date>_<time>/; log: results/clear27_single/logs/single.log.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
: "${1:?usage: run_single.sh sysadmin|sepsis|drone|aircraft \"<method>\" [--size S ...] [--n N ...] [--seeds SEED ...]}"
for arg in "$@"; do
    if [ "$arg" = "--list" ] || [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
        exec "$PY" experiments/clear27/run_single.py "$@"
    fi
done
[ -n "${CORE_SHARE:-}" ] && set -- "$@" --core-share "$CORE_SHARE"
LOG=results/clear27_single/logs
mkdir -p "$LOG"
if [ "${SHOW:-1}" = "1" ]; then
    "$PY" experiments/clear27/run_single.py "$@" 2>&1 | tee -a "$LOG/single.log"
    exit "${PIPESTATUS[0]}"
fi
"$PY" experiments/clear27/run_single.py "$@" >> "$LOG/single.log" 2>&1

#!/usr/bin/env bash
# clear27: one domain's 9 graph sources in ONE pool of workers (run_pool.py), from the repository root: a
# free worker takes the next (size, seed) job of any source, while each source keeps its waves by size and
# writes to its own folder. A stopped run continues with RESUME=1 (each source resumes its latest run
# folder, or starts one: RESUME=1 is also safe for a first launch).
#
#   bash experiments/clear27/run_domain.sh sysadmin             # every source of SysAdmin
#   bash experiments/clear27/run_domain.sh sysadmin flat pcmci  # only these sources
#   RESUME=1 bash experiments/clear27/run_domain.sh sysadmin    # continue the latest runs
#   SMOKE=1 bash experiments/clear27/run_domain.sh sysadmin     # tiny grids into results/clear27_smoke
#   CORE_SHARE=50 bash experiments/clear27/run_domain.sh sysadmin   # at most 50% of the physical cores
#   SHOW=0 ...                                                  # log only (default: also show the output)
# Logs: results/clear27[_smoke]/logs/ (chain.log: start, end and exit code of the pool; <domain>_pool.log).
set -u
cd "$(dirname "$0")/../.."
DOMAIN=${1:?usage: run_domain.sh sysadmin|sepsis|drone [source ...]}
shift
PY=${PY:-.venv/bin/python}
EXP=clear27
[ "${SMOKE:-0}" = "1" ] && EXP=clear27_smoke
LOG=results/$EXP/logs
mkdir -p "$LOG"
if [ "$#" -gt 0 ]; then
    SOURCES=("$@")
else
    SOURCES=(flat truegraph fullfact sparse dynotears_lagged pcmci_lagged sota met strehl)
fi
FLAGS=()
[ "${RESUME:-0}" = "1" ] && FLAGS=(--resume)
[ -n "${CORE_SHARE:-}" ] && FLAGS+=(--core-share "$CORE_SHARE")
ARGS=("$DOMAIN" --sources "${SOURCES[@]}" ${FLAGS[@]+"${FLAGS[@]}"})
if [ "${SHOW:-1}" = "1" ]; then
    "$PY" experiments/clear27/run_pool.py "${ARGS[@]}" 2>&1 | tee -a "$LOG/${DOMAIN}_pool.log"
    exit "${PIPESTATUS[0]}"
fi
"$PY" experiments/clear27/run_pool.py "${ARGS[@]}" >> "$LOG/${DOMAIN}_pool.log" 2>&1

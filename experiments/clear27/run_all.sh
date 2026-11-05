#!/usr/bin/env bash
# clear27: the four domains in ONE pool of workers (run_pool.py), SysAdmin first: a free worker takes a job of
# the first domain (sysadmin, drone, sepsis, aircraft) that has one waiting, so no worker idles while a domain
# waits for its slowest seed. RESUME=1, SMOKE=1, SHOW=0 as in run_domain.sh; log: logs/all_pool.log.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
EXP=clear27
[ "${SMOKE:-0}" = "1" ] && EXP=clear27_smoke
LOG=results/$EXP/logs
mkdir -p "$LOG"
FLAGS=()
[ "${RESUME:-0}" = "1" ] && FLAGS=(--resume)
ARGS=(sysadmin drone sepsis aircraft ${FLAGS[@]+"${FLAGS[@]}"})
if [ "${SHOW:-1}" = "1" ]; then
    "$PY" experiments/clear27/run_pool.py "${ARGS[@]}" 2>&1 | tee -a "$LOG/all_pool.log"
    exit "${PIPESTATUS[0]}"
fi
"$PY" experiments/clear27/run_pool.py "${ARGS[@]}" >> "$LOG/all_pool.log" 2>&1

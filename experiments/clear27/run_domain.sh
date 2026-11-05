#!/usr/bin/env bash
# clear27: one domain's 11 scripts in sequence (cheapest first), one process each, from the repository root;
# each script fills every worker with its own (size, seed) jobs. A script that fails is logged and the
# chain goes on; a stopped chain continues with RESUME=1 (each script resumes its latest run folder, or
# starts one: RESUME=1 is also safe for a first launch).
#
#   bash experiments/clear27/run_domain.sh sysadmin             # every source of SysAdmin
#   bash experiments/clear27/run_domain.sh sysadmin flat pcmci  # only these sources
#   RESUME=1 bash experiments/clear27/run_domain.sh sysadmin    # continue the latest runs
#   SMOKE=1 bash experiments/clear27/run_domain.sh sysadmin     # tiny grids into results/clear27_smoke
#   SHOW=0 ...                                                  # log only (default: also show the output)
# Logs: results/clear27[_smoke]/logs/ (chain.log: start, end and exit code of every script; <script>.log).
set -u
cd "$(dirname "$0")/../.."
DOMAIN=${1:?usage: run_domain.sh sysadmin|sepsis|drone|aircraft [source ...]}
shift
PY=${PY:-.venv/bin/python}
EXP=clear27
[ "${SMOKE:-0}" = "1" ] && EXP=clear27_smoke
LOG=results/$EXP/logs
mkdir -p "$LOG"
if [ "$#" -gt 0 ]; then
    SOURCES=("$@")
else  # cheapest first: a failure in the plumbing shows early
    SOURCES=(flat truegraph fullfact sparse dynotears_lagged dynotears pcmci_lagged pcmci sota met strehl)
fi
FLAGS=()
[ "${RESUME:-0}" = "1" ] && FLAGS=(--resume)

stamp() { date '+%F %T'; }
echo "$(stamp) $EXP $DOMAIN start: ${SOURCES[*]} (host $(hostname), commit $(git rev-parse --short HEAD 2>/dev/null))" \
    | tee -a "$LOG/chain.log"
for source in "${SOURCES[@]}"; do
    script="${DOMAIN}_${source}"
    [ -f "experiments/clear27/$script.py" ] || { echo "$(stamp) missing $script" | tee -a "$LOG/chain.log"; continue; }
    echo "$(stamp) start $script" | tee -a "$LOG/chain.log"
    start=$(date +%s)
    if [ "${SHOW:-1}" = "1" ]; then
        "$PY" "experiments/clear27/$script.py" ${FLAGS[@]+"${FLAGS[@]}"} 2>&1 | tee -a "$LOG/$script.log"
        rc=${PIPESTATUS[0]}
    else
        "$PY" "experiments/clear27/$script.py" ${FLAGS[@]+"${FLAGS[@]}"} >> "$LOG/$script.log" 2>&1
        rc=$?
    fi
    echo "$(stamp) end   $script rc=$rc ($(( ($(date +%s) - start) / 60 )) min)" | tee -a "$LOG/chain.log"
done
echo "$(stamp) $EXP $DOMAIN finished" | tee -a "$LOG/chain.log"

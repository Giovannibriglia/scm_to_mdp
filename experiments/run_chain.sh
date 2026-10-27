#!/usr/bin/env bash
# Run every script of one experiment in sequence (experiments/<exp>/<domain>_{imdp,mdp}.py), one process
# each, from the repository root. A grid that fails is logged and the chain goes on; a stopped chain can
# be started again with RESUME=1 (each grid continues its latest run folder).
#
#   bash experiments/run_chain.sh exp1                  # every domain of experiment 1, in the order below
#   bash experiments/run_chain.sh exp2 sysadmin taxi    # only these domains
#   RESUME=1 bash experiments/run_chain.sh exp1         # continue the latest runs
#   SHOW=0 bash experiments/run_chain.sh exp1           # log only (default: also show each script's output here)
# Logs: results/<exp>/logs/ (chain.log: start, end and exit code of every grid; <script>.log: its output).
set -u
cd "$(dirname "$0")/.."
EXP=${1:?usage: run_chain.sh exp1|exp2 [domain ...]}
shift
PY=${PY:-.venv/bin/python}
LOG=results/$EXP/logs
mkdir -p "$LOG"
if [ "$#" -gt 0 ]; then
    DOMAINS=("$@")
else
    DOMAINS=(sysadmin aircraft coffee drone frozenlake sepsis stock chemistry taxi)  # chemistry, taxi: slowest
fi
FLAGS=()
[ "${RESUME:-0}" = "1" ] && FLAGS=(--resume)

stamp() { date '+%F %T'; }
echo "$(stamp) $EXP chain start: ${DOMAINS[*]} (host $(hostname), commit $(git rev-parse --short HEAD 2>/dev/null))" \
    | tee -a "$LOG/chain.log"
for domain in "${DOMAINS[@]}"; do
    for kind in imdp mdp; do
        script="${domain}_${kind}"
        echo "$(stamp) start $script" | tee -a "$LOG/chain.log"
        start=$(date +%s)
        if [ "${SHOW:-1}" = "1" ]; then  # to the terminal (progress bar) and the log
            "$PY" "experiments/$EXP/$script.py" ${FLAGS[@]+"${FLAGS[@]}"} 2>&1 | tee -a "$LOG/$script.log"
            rc=${PIPESTATUS[0]}  # the script's exit code, not tee's
        else  # to the log only
            "$PY" "experiments/$EXP/$script.py" ${FLAGS[@]+"${FLAGS[@]}"} >> "$LOG/$script.log" 2>&1
            rc=$?
        fi
        echo "$(stamp) end   $script rc=$rc ($(( ($(date +%s) - start) / 60 )) min)" | tee -a "$LOG/chain.log"
    done
done
echo "$(stamp) $EXP chain finished" | tee -a "$LOG/chain.log"

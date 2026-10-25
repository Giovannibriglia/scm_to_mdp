#!/usr/bin/env bash
# Run every trial grid in sequence (experiments/trials/<domain>_uniform_{rmdp,mdp}.py), one process each,
# from the repository root. A grid that fails is logged and the chain goes on; a stopped chain can be
# started again with RESUME=1 (each grid continues its latest run folder).
#
#   bash experiments/trials/run_chain.sh                 # every domain, in the order below
#   bash experiments/trials/run_chain.sh sysadmin taxi   # only these domains
#   RESUME=1 bash experiments/trials/run_chain.sh        # continue the latest runs
# Logs: results/trials/logs/ (chain.log: start, end and exit code of every grid; <script>.log: its output).
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
LOG=results/trials/logs
mkdir -p "$LOG"
if [ "$#" -gt 0 ]; then
    DOMAINS=("$@")
else
    DOMAINS=(sysadmin aircraft coffee drone frozenlake sepsis stock chemistry taxi)  # chemistry, taxi: slowest
fi
FLAGS=()
[ "${RESUME:-0}" = "1" ] && FLAGS=(--resume)

stamp() { date '+%F %T'; }
echo "$(stamp) chain start: ${DOMAINS[*]} (host $(hostname), commit $(git rev-parse --short HEAD 2>/dev/null))" \
    | tee -a "$LOG/chain.log"
for domain in "${DOMAINS[@]}"; do
    for kind in rmdp mdp; do
        script="${domain}_uniform_${kind}"
        echo "$(stamp) start $script" | tee -a "$LOG/chain.log"
        start=$(date +%s)
        "$PY" "experiments/trials/$script.py" ${FLAGS[@]+"${FLAGS[@]}"} >> "$LOG/$script.log" 2>&1
        rc=$?
        echo "$(stamp) end   $script rc=$rc ($(( ($(date +%s) - start) / 60 )) min)" | tee -a "$LOG/chain.log"
    done
done
echo "$(stamp) chain finished" | tee -a "$LOG/chain.log"

#!/usr/bin/env bash
# ablation_rewards: one domain, the run and then its figures, from the repository root.
#
#   bash experiments/clear27/ablations/ablation_rewards/run_domain.sh sysadmin   # or run_sysadmin.sh
#   RESUME=1 bash .../run_domain.sh sepsis          # continue the latest run of the domain
#   SIZES="3 4" SEEDS=2 bash .../run_domain.sh drone   # fewer sizes (machines, vitals, side) and seeds
#   TRANSITIONS="100 1000" ...   WORKERS=8 ...   CORE_SHARE=50 ...   PLOT=0 (the run only)
#
# Defaults: the sizes of run.py (SysAdmin N = 3..8 here, as in the paper; Sepsis 2 3 4 vitals; Drone side
# 3..9), 10 seeds, 10 ... 100,000 transitions.
# Results: results/clear27/ablations/ablation_rewards/<domain>/<domain>_<date>_<time>/ (launch.log beside it)
# Figures: outputs/clear27/ablations/ablation_rewards/<domain>/
set -u
cd "$(dirname "$0")/../../../.."
DOMAIN=${1:?usage: run_domain.sh sysadmin|sepsis|drone}
PY=${PY:-.venv/bin/python}
HERE=experiments/clear27/ablations/ablation_rewards
OUT=results/clear27/ablations/ablation_rewards/$DOMAIN
mkdir -p "$OUT"
TRANSITIONS=${TRANSITIONS:-10 100 1000 10000 20000 50000 100000}
[ "$DOMAIN" = "sysadmin" ] && SIZES=${SIZES:-3 4 5 6 7 8}
# shellcheck disable=SC2206
ARGS=(--domain "$DOMAIN" --transitions $TRANSITIONS)
# shellcheck disable=SC2206
[ -n "${SIZES:-}" ] && ARGS+=(--sizes $SIZES)
[ -n "${SEEDS:-}" ] && ARGS+=(--seeds "$SEEDS")
[ -n "${WORKERS:-}" ] && ARGS+=(--workers "$WORKERS")
[ -n "${CORE_SHARE:-}" ] && ARGS+=(--core-share "$CORE_SHARE")
[ "${RESUME:-0}" = "1" ] && ARGS+=(--resume)
echo "$(date '+%F %T') start $DOMAIN: ${ARGS[*]}" | tee -a "$OUT/launch.log"
"$PY" "$HERE/run.py" "${ARGS[@]}" 2>&1 | tee -a "$OUT/launch.log"
rc=${PIPESTATUS[0]}
echo "$(date '+%F %T') end   $DOMAIN rc=$rc" | tee -a "$OUT/launch.log"
if [ "${PLOT:-1}" = "1" ]; then
    "$PY" "$HERE/plot.py" --domain "$DOMAIN" 2>&1 | tee -a "$OUT/launch.log"
fi
exit "$rc"

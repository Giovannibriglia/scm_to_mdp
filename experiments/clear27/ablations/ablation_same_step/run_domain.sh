#!/usr/bin/env bash
# ablation_same_step: one domain, the run and then the figures, from the repository root.
#
#   bash experiments/clear27/ablations/ablation_same_step/run_domain.sh sysadmin   # or run_sysadmin.sh
#   RESUME=1 bash .../run_domain.sh sepsis             # continue the latest runs of the domain
#   SIZES="3 4" SEEDS="0 1" bash .../run_domain.sh sysadmin   # fewer sizes and seeds (values of the paper's grid)
#   TRANSITIONS="100 1000" ...   SOURCES="pcmci" ...   WORKERS=8 ...   CORE_SHARE=50 ...
#   EVERY=1 ...        every compilation of clear27 on the two graphs (default: the box IMDP and the plug-in MDP)
#   LAGGED=1 ...       first run the twins without same-step arrows, which are sources of the main runs
#                      (experiments/clear27/run_domain.sh <domain> dynotears_lagged pcmci_lagged)
#   PLOT=0 ...         the run only
#
# Defaults: the paper's sizes, 10 seeds and 15 numbers of transitions (protocol.py).
# Results: results/clear27/ablations/ablation_same_step/<domain>/<domain>_<source>_<date>_<time>/
# Figures: outputs/clear27/ablations/ablation_same_step/<domain>/ and all_domains/
set -u
cd "$(dirname "$0")/../../../.."
DOMAIN=${1:?usage: run_domain.sh sysadmin|sepsis|drone}
PY=${PY:-.venv/bin/python}
HERE=experiments/clear27/ablations/ablation_same_step
OUT=results/clear27/ablations/ablation_same_step/$DOMAIN
mkdir -p "$OUT"
ARGS=("$DOMAIN")
# shellcheck disable=SC2206
[ -n "${SOURCES:-}" ] && ARGS+=(--sources $SOURCES)
# shellcheck disable=SC2206
[ -n "${SIZES:-}" ] && ARGS+=(--sizes $SIZES)
# shellcheck disable=SC2206
[ -n "${TRANSITIONS:-}" ] && ARGS+=(--transitions $TRANSITIONS)
# shellcheck disable=SC2206
[ -n "${SEEDS:-}" ] && ARGS+=(--seeds $SEEDS)
[ -n "${WORKERS:-}" ] && ARGS+=(--workers "$WORKERS")
[ -n "${CORE_SHARE:-}" ] && ARGS+=(--core-share "$CORE_SHARE")
[ "${EVERY:-0}" = "1" ] && ARGS+=(--every-compilation)
[ "${RESUME:-0}" = "1" ] && ARGS+=(--resume)
if [ "${LAGGED:-0}" = "1" ]; then
    bash experiments/clear27/run_domain.sh "$DOMAIN" dynotears_lagged pcmci_lagged || exit $?
fi
echo "$(date '+%F %T') start $DOMAIN: ${ARGS[*]}" | tee -a "$OUT/launch.log"
"$PY" "$HERE/run.py" "${ARGS[@]}" 2>&1 | tee -a "$OUT/launch.log"
rc=${PIPESTATUS[0]}
echo "$(date '+%F %T') end   $DOMAIN rc=$rc" | tee -a "$OUT/launch.log"
if [ "${PLOT:-1}" = "1" ]; then
    "$PY" "$HERE/plot.py" "$DOMAIN" 2>&1 | tee -a "$OUT/launch.log"
fi
exit "$rc"

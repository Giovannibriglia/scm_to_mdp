#!/usr/bin/env bash
# ablation_box_exact: SysAdmin, Drone and Sepsis in one pool of workers, from the repository root (run.py; no figures).
#   RESUME=1 ...   WORKERS=N ...   CORE_SHARE=P ...      as in run.py (--resume, --workers, --core-share)
# Results: results/clear27/ablations/ablation_box_exact/<domain>/; log: results/clear27/ablations/ablation_box_exact/all.log
# Figures, after the run: .venv/bin/python experiments/clear27/ablations/ablation_box_exact/plot.py
set -u
cd "$(dirname "$0")/../../../.."
PY=${PY:-.venv/bin/python}
OUT=results/clear27/ablations/ablation_box_exact
mkdir -p "$OUT"
ARGS=(sysadmin drone sepsis)
[ "${RESUME:-0}" = "1" ] && ARGS+=(--resume)
[ -n "${WORKERS:-}" ] && ARGS+=(--workers "$WORKERS")
[ -n "${CORE_SHARE:-}" ] && ARGS+=(--core-share "$CORE_SHARE")
"$PY" experiments/clear27/ablations/ablation_box_exact/run.py "${ARGS[@]}" 2>&1 | tee -a "$OUT/all.log"
exit "${PIPESTATUS[0]}"

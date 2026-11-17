#!/usr/bin/env bash
# ablation_rewards: SysAdmin, Drone and Sepsis one after the other, from the repository root (run_domain.sh
# with PLOT=0; no figures). RESUME=1, WORKERS=N, CORE_SHARE=P, SEEDS=K as in run_domain.sh.
# Results: results/clear27/ablations/ablation_rewards/<domain>/
# Figures, after the run: .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py --domain all_domains
set -u
cd "$(dirname "$0")/../../../.."
rc=0
for domain in sysadmin drone sepsis; do
    PLOT=0 bash experiments/clear27/ablations/ablation_rewards/run_domain.sh "$domain" || rc=$?
done
exit "$rc"

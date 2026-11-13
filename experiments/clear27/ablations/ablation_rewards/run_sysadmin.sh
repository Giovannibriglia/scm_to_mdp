#!/usr/bin/env bash
# ablation_rewards, sysadmin (N = 3..8): the run, then its figures (run_domain.sh: RESUME=1, SIZES, SEEDS, ...).
exec bash "$(dirname "$0")/run_domain.sh" sysadmin "$@"

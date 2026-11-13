#!/usr/bin/env bash
# ablation_rewards, drone (side 3..9): the run, then its figures (run_domain.sh: RESUME=1, SIZES, SEEDS, ...).
exec bash "$(dirname "$0")/run_domain.sh" drone "$@"

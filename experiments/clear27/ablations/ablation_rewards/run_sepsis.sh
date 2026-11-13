#!/usr/bin/env bash
# ablation_rewards, sepsis (2, 3, 4 vitals): the run, then its figures (run_domain.sh: RESUME=1, SIZES, SEEDS, ...).
exec bash "$(dirname "$0")/run_domain.sh" sepsis "$@"

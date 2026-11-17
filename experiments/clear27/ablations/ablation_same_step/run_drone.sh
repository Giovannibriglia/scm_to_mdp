#!/usr/bin/env bash
# ablation_same_step, drone: the run, then its figures (run_domain.sh: RESUME=1, SIZES, SEEDS, LAGGED=1, ...).
exec bash "$(dirname "$0")/run_domain.sh" drone "$@"

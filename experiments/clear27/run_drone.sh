#!/usr/bin/env bash
# clear27, drone: its 11 sources in sequence (run_domain.sh: RESUME=1, SMOKE=1, SHOW=0, a list of sources).
exec bash "$(dirname "$0")/run_domain.sh" drone "$@"

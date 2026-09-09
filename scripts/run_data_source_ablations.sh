#!/usr/bin/env bash
set -euo pipefail
# v2: bind each source to its own trained adapters, feature models and calibration.
if [[ $# -lt 1 ]]; then
  echo "usage: $0 BASELINE_CHECKPOINT [--mode pilot|full] [--stages all|controls|evaluate] ..." >&2
  exit 2
fi
baseline="$1"
shift
for source in iemocap emotiontalk; do
  python -B scripts/run_v2_experiments.py --source "$source" --baseline "$baseline" "$@"
done

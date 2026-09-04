#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 TRAIN_PID" >&2
  exit 2
fi

train_pid="$1"
while kill -0 "$train_pid" 2>/dev/null; do
  sleep 60
done

checkpoint="runs/dualtalk_baseline_control/equal_epoch3/last.pt"
if [[ ! -f "$checkpoint" ]]; then
  echo "baseline control checkpoint was not produced: $checkpoint" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

python -B -m emotion_ssm.evaluate_dualtalk \
  --config configs/dualtalk_conditioned.yaml \
  --model baseline \
  --checkpoint "$checkpoint" \
  --split test \
  --output runs/comparison_epoch3/baseline_equal_epoch3_test.json

python -B -m emotion_ssm.evaluate_dualtalk \
  --config configs/dualtalk_conditioned.yaml \
  --model baseline \
  --checkpoint "$checkpoint" \
  --split ood \
  --output runs/comparison_epoch3/baseline_equal_epoch3_ood.json


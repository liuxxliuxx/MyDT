#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 IEMOCAP_DYNAMICS_PID" >&2
  exit 2
fi

initial_pid="$1"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

while kill -0 "$initial_pid" 2>/dev/null; do
  sleep 60
done

if ! tail -n 1 runs/phase_a_dynamics/iemocap_only/metrics.jsonl | grep -q '"epoch": 50'; then
  echo "IEMOCAP-only Phase A dynamics did not reach epoch 50" >&2
  exit 1
fi

run_phase_b() {
  local source="$1"
  python -B -m emotion_ssm.train.phase_b \
    --config configs/phase_b.yaml \
    DEVICE cuda:0 \
    TRAIN.EXPERIMENT_NAME "${source}_only" \
    DATA.SOURCES "['${source}']" \
    TRAIN.GRAD_ACCUMULATION 2 \
    TRAIN.OBSERVATION_CHECKPOINT "runs/phase_a_observation/${source}_only/observation_encoder.pt" \
    TRAIN.EMA_TEACHER_CHECKPOINT "runs/phase_a_observation/${source}_only/ema_teacher.pt" \
    TRAIN.EMOTION_HEADS_CHECKPOINT "runs/phase_a_observation/${source}_only/emotion_heads.pt" \
    TRAIN.PHASE_A_CHECKPOINT "runs/phase_a_dynamics/${source}_only/phase_a_best.pt" \
    > "phase_b_${source}_only.log" 2>&1
}

run_dualtalk() {
  local source="$1"
  python -B -m emotion_ssm.train.dualtalk \
    --config configs/dualtalk_conditioned.yaml \
    DEVICE cuda:0 \
    DATA.SOURCES "['${source}']" \
    TRAIN.EXPERIMENT_NAME "${source}_only_epoch3" \
    TRAIN.GRAD_ACCUMULATION 2 \
    TRAIN.STOP_AFTER_EPOCHS 3 \
    TRAIN.OBSERVATION_CHECKPOINT "runs/phase_a_observation/${source}_only/observation_encoder.pt" \
    TRAIN.PHASE_B_CHECKPOINT "runs/phase_b_coupling/${source}_only/phase_b_best.pt" \
    DUALTALK.PHASE_B_CHECKPOINT "runs/phase_b_coupling/${source}_only/phase_b_best.pt" \
    DUALTALK.BASELINE_CHECKPOINT model/dualtalk_baseline.pth \
    > "dualtalk_${source}_only_epoch3.log" 2>&1
}

evaluate_source() {
  local source="$1"
  local checkpoint="runs/dualtalk_conditioned/${source}_only_epoch3/dualtalk_emotion_best.pt"
  local output_dir="runs/comparison_data_sources"
  python -B -m emotion_ssm.evaluate_dualtalk \
    --config configs/dualtalk_conditioned.yaml \
    --model conditioned \
    --ablation full \
    --checkpoint "$checkpoint" \
    --baseline-checkpoint model/dualtalk_baseline.pth \
    --split test \
    --output "${output_dir}/${source}_only_test.json"
  python -B -m emotion_ssm.evaluate_dualtalk \
    --config configs/dualtalk_conditioned.yaml \
    --model conditioned \
    --ablation full \
    --checkpoint "$checkpoint" \
    --baseline-checkpoint model/dualtalk_baseline.pth \
    --split ood \
    --output "${output_dir}/${source}_only_ood.json"
}

run_phase_b iemocap
run_dualtalk iemocap
evaluate_source iemocap

python -B -m emotion_ssm.train.phase_a_observation \
  --config configs/phase_a_observation.yaml \
  DEVICE cuda:0 \
  TRAIN.EXPERIMENT_NAME emotiontalk_only \
  DATA.SOURCES "['emotiontalk']" \
  TRAIN.GRAD_ACCUMULATION 2 \
  > phase_a_observation_emotiontalk_only.log 2>&1

python -B -m emotion_ssm.train.phase_a_dynamics \
  --config configs/phase_a_dynamics.yaml \
  DEVICE cuda:0 \
  TRAIN.EXPERIMENT_NAME emotiontalk_only \
  DATA.SOURCES "['emotiontalk']" \
  TRAIN.GRAD_ACCUMULATION 2 \
  TRAIN.OBSERVATION_CHECKPOINT runs/phase_a_observation/emotiontalk_only/observation_encoder.pt \
  TRAIN.EMA_TEACHER_CHECKPOINT runs/phase_a_observation/emotiontalk_only/ema_teacher.pt \
  TRAIN.EMOTION_HEADS_CHECKPOINT runs/phase_a_observation/emotiontalk_only/emotion_heads.pt \
  > phase_a_dynamics_emotiontalk_only.log 2>&1

run_phase_b emotiontalk
run_dualtalk emotiontalk
evaluate_source emotiontalk


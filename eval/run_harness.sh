#!/usr/bin/env bash
# run_harness.sh — Wrapper around lm-evaluation-harness
#
# Usage:
#   bash eval/run_harness.sh <checkpoint_path> [task_list] [output_path]
#
# Prerequisites:
#   pip install lm-eval
#
# Example:
#   bash eval/run_harness.sh ./checkpoints/step_200000 \
#       "mmlu,hellaswag,arc_challenge,gsm8k" \
#       eval/results/step_200000.json

set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <checkpoint_path> [tasks] [output_path]}"
TASKS="${2:-mmlu,hellaswag,arc_challenge,gsm8k,humaneval}"
OUTPUT="${3:-eval/results/results.json}"
BATCH_SIZE="${4:-16}"

mkdir -p "$(dirname "$OUTPUT")"

echo "Running lm-evaluation-harness"
echo "  Checkpoint : $CHECKPOINT"
echo "  Tasks      : $TASKS"
echo "  Output     : $OUTPUT"
echo "  Batch size : $BATCH_SIZE"

lm_eval \
    --model hf \
    --model_args "pretrained=$CHECKPOINT" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT" \
    --log_samples

echo "Evaluation complete. Results written to $OUTPUT"

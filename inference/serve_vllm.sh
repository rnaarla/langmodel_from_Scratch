#!/usr/bin/env bash
# serve_vllm.sh — Launch vLLM OpenAI-compatible API server
#
# Prerequisites:
#   pip install vllm
#
# Usage:
#   bash inference/serve_vllm.sh [MODEL_PATH] [QUANTIZATION] [TENSOR_PARALLEL_SIZE]
#
# Examples:
#   bash inference/serve_vllm.sh ./checkpoints/final
#   bash inference/serve_vllm.sh ./model_awq awq 2

set -euo pipefail

MODEL_PATH="${1:-./checkpoints/final}"
QUANTIZATION="${2:-}"        # awq | gptq | squeezellm | (empty = none)
TP_SIZE="${3:-1}"            # tensor parallel size (number of GPUs)
MAX_MODEL_LEN="${4:-8192}"
PORT="${5:-8000}"

echo "Starting vLLM server"
echo "  Model      : $MODEL_PATH"
echo "  Quant      : ${QUANTIZATION:-none}"
echo "  TP size    : $TP_SIZE"
echo "  Max seq len: $MAX_MODEL_LEN"
echo "  Port       : $PORT"
echo ""

QUANT_ARG=""
if [[ -n "$QUANTIZATION" ]]; then
    QUANT_ARG="--quantization $QUANTIZATION"
fi

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    $QUANT_ARG \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name llm-from-scratch

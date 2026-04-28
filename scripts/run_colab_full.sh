#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -z "${PYTHON:-}" && -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi

# Qwen2.5-3B is the default Colab GPU run. Override settings with env vars:
#   MODEL_NAME=Qwen/Qwen2.5-7B LOAD_IN_4BIT=true OUTPUT_DIR=results/colab_qwen7b scripts/run_colab_full.sh
#   MODEL_NAME=Qwen/Qwen2.5-1.5B scripts/run_colab_full.sh
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B}"
SEQ_LENS="${SEQ_LENS:-128 256 512 1024}"
GENERATED_TOKENS="${GENERATED_TOKENS:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-results/colab_full}"
ACCURACY_STEPS="${ACCURACY_STEPS:-16}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ "$LOAD_IN_4BIT" == "true" ]]; then
  EXTRA_ARGS+=(--load-in-4bit)
fi

"$PYTHON_BIN" benchmarks/benchmark.py \
  --model-name "$MODEL_NAME" \
  --seq-lens $SEQ_LENS \
  --generated-tokens "$GENERATED_TOKENS" \
  --batch-size 1 \
  --block-size 16 \
  --run-adaptive-ablations true \
  --measure-accuracy true \
  --accuracy-steps "$ACCURACY_STEPS" \
  --adaptive-monotonic true \
  --device cuda \
  --output-dir "$OUTPUT_DIR" \
  --notes "Colab full adaptive KV benchmark" \
  "${EXTRA_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -z "${PYTHON:-}" && -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi
mkdir -p results/colab_full

# Qwen2.5-3B is the intended Colab GPU run. If memory is tight, use:
#   MODEL_NAME=Qwen/Qwen2.5-1.5B scripts/run_colab_full.sh
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B}"

"$PYTHON_BIN" benchmarks/benchmark.py \
  --model-name "$MODEL_NAME" \
  --seq-lens 128 256 512 1024 \
  --generated-tokens 64 \
  --batch-size 1 \
  --block-size 16 \
  --run-adaptive-ablations true \
  --measure-accuracy true \
  --accuracy-steps 16 \
  --adaptive-monotonic true \
  --device cuda \
  --output-dir results/colab_full \
  --notes "Colab full adaptive KV benchmark"

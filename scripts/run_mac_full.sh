#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -z "${PYTHON:-}" && -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi
mkdir -p results/mac_full

# Qwen2.5-3B may be heavy on Mac MPS. Override with:
#   MODEL_NAME=Qwen/Qwen2.5-1.5B scripts/run_mac_full.sh
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B}"

"$PYTHON_BIN" benchmarks/benchmark.py \
  --model-name "$MODEL_NAME" \
  --seq-lens 128 256 512 \
  --generated-tokens 32 \
  --batch-size 1 \
  --block-size 16 \
  --run-adaptive-ablations true \
  --measure-accuracy true \
  --accuracy-steps 16 \
  --adaptive-monotonic true \
  --device auto \
  --output-dir results/mac_full \
  --notes "macOS full adaptive KV benchmark"

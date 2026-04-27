#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -z "${PYTHON:-}" && -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python3}"
fi
mkdir -p results/mac_smoke

"$PYTHON_BIN" benchmarks/benchmark.py \
  --model-name sshleifer/tiny-gpt2 \
  --sequence-length 64 \
  --generated-tokens 8 \
  --batch-size 1 \
  --block-size 16 \
  --mode baseline,int8,blocked_int8,adaptive_recency_k16 \
  --measure-accuracy true \
  --accuracy-steps 8 \
  --adaptive-monotonic true \
  --device auto \
  --output-dir results/mac_smoke \
  --notes "macOS smoke benchmark"

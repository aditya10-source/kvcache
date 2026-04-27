#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" benchmarks/benchmark.py \
  --model "${MODEL:-sshleifer/tiny-gpt2}" \
  --seq-lens 128 256 \
  --max-new-tokens 16 \
  --block-size 16 \
  --output-dir results

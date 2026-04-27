$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force -Path "results/windows_smoke" | Out-Null

python benchmarks/benchmark.py `
  --model-name sshleifer/tiny-gpt2 `
  --sequence-length 64 `
  --generated-tokens 8 `
  --batch-size 1 `
  --block-size 16 `
  --mode baseline,int8,blocked_int8,adaptive_recency_k16 `
  --measure-accuracy true `
  --accuracy-steps 8 `
  --adaptive-monotonic true `
  --device auto `
  --output-dir results/windows_smoke `
  --notes "Windows smoke benchmark"

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force -Path "results/windows_full" | Out-Null

if (-not $env:MODEL_NAME) {
  $env:MODEL_NAME = "Qwen/Qwen2.5-1.5B"
}

python benchmarks/benchmark.py `
  --model-name $env:MODEL_NAME `
  --seq-lens 128 256 512 `
  --generated-tokens 32 `
  --batch-size 1 `
  --block-size 16 `
  --run-adaptive-ablations true `
  --measure-accuracy true `
  --accuracy-steps 16 `
  --adaptive-monotonic true `
  --device auto `
  --output-dir results/windows_full `
  --notes "Windows full adaptive KV benchmark"

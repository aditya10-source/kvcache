# KV Cache Quantization and Block-Based Layout

Class project codebase for benchmarking faster autoregressive Transformer decoding with:

1. Baseline floating-point KV cache
2. INT8 quantized KV cache
3. INT8 quantized KV cache with blocked layout `[B, H, Nb, Bs, D]`
4. Adaptive hybrid KV cache with per-block FP16/INT8/INT4 precision

The implementation is intentionally simple and educational. Hugging Face models still consume floating-point `past_key_values`, so the INT8 and adaptive experiments quantize the cache after each decode step and dequantize it before the next step. This measures cache memory footprint, compression policy behavior, and quantization overhead in a runnable class-project setting, while leaving custom attention-kernel integration as a TODO.

## References and Borrowed Ideas

No source code is copied from these projects. The code here uses public APIs and reimplements the project logic from scratch.

- Hugging Face Transformers KV-cache documentation: https://huggingface.co/docs/transformers/en/kv_cache
- Hugging Face generation/cache API concepts: https://huggingface.co/docs/transformers/v4.55.4/kv_cache
- vLLM PagedAttention/block-based KV-cache concept: https://docs.vllm.ai/en/v0.18.0/design/paged_attention/
- vLLM conceptual docs for block-based KV storage: https://www.mintlify.com/vllm-project/vllm/concepts/paged-attention

## Repo Structure

```text
.
├── README.md
├── requirements.txt
├── src/kv_cache_quant/
│   ├── adaptive_kv_policy.py
│   ├── baseline_decode.py
│   ├── blocked_kv_cache.py
│   ├── importance_score.py
│   ├── kv_quant.py
│   ├── similarity.py
│   └── utils.py
├── benchmarks/
│   └── benchmark.py
├── scripts/
│   ├── run_benchmark.sh
│   └── smoke_test.py
├── notebooks/
│   └── colab_demo.ipynb
└── results/
```

## Installation

### macOS CPU/MPS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-mac.txt
```

PyTorch will use CUDA if available, then MPS on Apple Silicon, then CPU.

### Google Colab or Kaggle

```bash
!git clone <your-repo-url>
%cd <your-repo-name>
!pip install -r requirements.txt
```

Then run:

```bash
!python benchmarks/benchmark.py --model sshleifer/tiny-gpt2 --seq-lens 128 256 --max-new-tokens 16
```

## Quick Smoke Test

This test validates the quantization, blocked-layout, and adaptive-cache round trips without downloading a model:

```bash
python scripts/smoke_test.py
```

## Run Benchmarks

Small functionality run:

```bash
python benchmarks/benchmark.py \
  --model sshleifer/tiny-gpt2 \
  --seq-lens 128 256 \
  --max-new-tokens 16 \
  --block-size 16 \
  --output-dir results
```

Default Qwen run:

```bash
python benchmarks/benchmark.py \
  --model Qwen/Qwen2.5-1.5B \
  --seq-lens 128 256 512 \
  --max-new-tokens 32 \
  --block-size 16 \
  --output-dir results
```

DistilGPT-2 run:

```bash
python benchmarks/benchmark.py \
  --model distilgpt2 \
  --seq-lens 128 256 512 992 \
  --max-new-tokens 32 \
  --block-size 16
```

For GPT-2-style models with a 1024-token context limit, `seq_len + max_new_tokens` must fit within the context window. The benchmark detects this and lowers the effective prompt length while preserving `requested_seq_len` in the CSV/JSON output. Use a longer-context model, such as TinyLlama, for full 1024-token prompt experiments.

Outputs are written to:

- `results/benchmark_results.csv`
- `results/benchmark_results.json`
- `results/benchmark_tokens_per_sec.png` if matplotlib is installed


## Adaptive Hybrid KV Cache

The adaptive mode is a lightweight inference-time policy inspired by recent adaptive KV-cache research ideas. It is not copied from a paper implementation. Each sequence block receives one precision assignment:

- `FP16` for the most important blocks
- `INT8` for medium-importance blocks
- `INT4` for low-importance blocks

Importance scoring supports three modes:

- `recency`: newer blocks are more important
- `attention`: blocks receiving more attention mass are more important
- `hybrid`: `alpha * recency + beta * attention`

Example:

```bash
python benchmarks/benchmark.py \
  --model Qwen/Qwen2.5-1.5B \
  --seq-lens 128 256 512 \
  --max-new-tokens 32 \
  --modes baseline int8 blocked_int8 adaptive \
  --importance-mode hybrid \
  --alpha 0.5 \
  --beta 0.5 \
  --fp16-ratio 0.2 \
  --int8-ratio 0.5
```

For Qwen2.5-3B, use:

```bash
python benchmarks/benchmark.py --model Qwen/Qwen2.5-3B --seq-lens 128 256 512 --max-new-tokens 32
```

For Mistral-7B on a CUDA machine with bitsandbytes installed, optional 4-bit model weights are supported:

```bash
python benchmarks/benchmark.py \
  --model mistralai/Mistral-7B-v0.1 \
  --load-in-4bit \
  --seq-lens 128 256 \
  --max-new-tokens 16
```

## What the Metrics Mean

- `tokens_per_sec`: generated tokens divided by measured decode wall time
- `latency_ms_per_token`: average wall-clock latency per generated token
- `kv_cache_mb`: estimated KV-cache memory footprint for the final cache
- `compression_ratio`: FP16 baseline KV bytes divided by the current mode's estimated KV bytes
- `token_match_rate`: generated-token agreement versus the baseline run
- `mean_abs_logit_error`: average absolute error between step logits and baseline logits
- `cosine_similarity`: cosine similarity between final-step logits and baseline final-step logits

## Design Notes

- Baseline uses the model's native `past_key_values`.
- INT8 mode uses symmetric signed INT8 quantization with per-block scales along the sequence dimension.
- Blocked INT8 mode stores quantized tensors as `[B, H, Nb, Bs, D]` and unblocks them before dequantizing back to Hugging Face's expected `[B, H, S, D]`.
- Adaptive mode stores each sequence block as FP16, INT8, or simulated packed INT4 according to recency, attention, or hybrid importance. INT4 values are represented in `int8` tensors for portability, but memory accounting assumes two 4-bit values per byte.
- Batch size defaults to 1 because the proposal focuses on single-request decoding.

## TODOs for Deeper Systems Work

- Integrate quantized KV tensors directly into attention kernels instead of dequantizing before each Hugging Face forward call.
- Add a true paged/block table allocator for non-contiguous cache blocks.
- Add CUDA kernels or Triton kernels for dequantize-and-attend fusion.
- Compare against Hugging Face's built-in quantized cache implementations when available for the chosen model/version.
- Add profiler traces for memory bandwidth and kernel timings on GPU.

## Enhanced Benchmarking and Adaptive Ablations

The benchmark runner now saves detailed results for every run. It keeps the original manual decode loop and does not call `generate()`.

Baseline only:

```bash
python benchmarks/benchmark.py \
  --model-name sshleifer/tiny-gpt2 \
  --sequence-length 128 \
  --generated-tokens 16 \
  --mode baseline \
  --output-dir results
```

All core modes:

```bash
python benchmarks/benchmark.py \
  --model-name Qwen/Qwen2.5-1.5B \
  --seq-lens 128 256 512 \
  --generated-tokens 32 \
  --mode baseline,int8,blocked_int8,adaptive \
  --adaptive-policy recency \
  --adaptive-update-interval 16 \
  --adaptive-monotonic true \
  --measure-accuracy true \
  --output-dir results
```

Adaptive ablations:

```bash
python benchmarks/benchmark.py \
  --model-name Qwen/Qwen2.5-1.5B \
  --seq-lens 128 256 512 \
  --generated-tokens 32 \
  --run-adaptive-ablations true \
  --measure-accuracy true \
  --output-dir results
```

The ablation set includes:

- `adaptive_recency_k1`
- `adaptive_recency_k16`
- `adaptive_recency_k32`
- `adaptive_attention_k16`
- `adaptive_hybrid_k16`

Attention and hybrid policies require `output_attentions=True`, so they are expected to be slower. Recency adaptive mode avoids attention extraction and is the fast adaptive path.

Saved result files:

- `results/summary_runs.csv`
- `results/runs/{run_id}.json`
- `results/raw/{run_id}_timing.csv`
- `results/raw/{run_id}_accuracy.csv`
- legacy compatibility outputs: `results/benchmark_results.csv` and `results/benchmark_results.json`

Plot saved summaries:

```bash
python benchmarks/plot_results.py \
  --summary results/summary_runs.csv \
  --output-dir results/plots
```

Benchmark smoke test:

```bash
python scripts/smoke_benchmark.py
```

This runs a tiny model with sequence length 64, 8 generated tokens, and modes `baseline`, `int8`, `blocked_int8`, and `adaptive_recency_k16`. It verifies that result files exist and accuracy metrics are finite.

## Why Adaptive Can Be Slower

This project preserves honest measurements. The adaptive cache policy runs at Python level: it scores blocks, assigns precision, quantizes cache tensors, and dequantizes them back into Hugging Face-compatible floating-point `past_key_values`. That extra work can outweigh memory savings, especially for small models or short sequences.

The timing breakdown in each per-run JSON and raw timing CSV helps explain the slowdown. For adaptive mode, it records:

- total decode time
- model forward time
- importance scoring time
- quantization time
- dequantization time
- precision assignment time
- tensor copy/re-layout time

Memory savings and accuracy tradeoffs remain meaningful, but improving tokens/sec in a production system would require fused CUDA/Triton attention kernels that consume mixed-precision KV blocks directly.

Note on `--adaptive-monotonic`: because Hugging Face attention still requires floating-point caches, this prototype must dequantize before each model forward. The monotonic option prevents precision assignments from upgrading or oscillating across adaptive updates; it does not remove the dequantization step that would require a custom fused attention kernel.

# Reproducible Platform Runs

The benchmark supports `--device auto`, which chooses CUDA first, then Apple MPS if available, then CPU. On Windows, MPS is not available, so `auto` chooses CUDA if `torch.cuda.is_available()` is true and CPU otherwise. The selected device is printed at runtime in the `Loading ... on <device>` line.

## Run On A Mac

Use this section for a fresh Apple Silicon or CPU-only Mac. Run all commands from the repo root. Do not install CUDA-only packages on Mac.

1. Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-mac.txt
```

2. Run the local smoke test. This does not download a model:

```bash
python scripts/smoke_test.py
```

3. Run a small benchmark with Tiny GPT-2. This downloads a small Hugging Face model and writes results to `results/tiny_gpt2`:

```bash
python benchmarks/benchmark.py \
  --model-name sshleifer/tiny-gpt2 \
  --seq-lens 128 256 \
  --generated-tokens 16 \
  --batch-size 1 \
  --block-size 16 \
  --mode baseline,int8,blocked_int8,adaptive \
  --measure-accuracy true \
  --accuracy-steps 16 \
  --device auto \
  --output-dir results/tiny_gpt2
```

4. Generate plots for that run:

```bash
python benchmarks/plot_results.py \
  --summary results/tiny_gpt2/summary_runs.csv \
  --output-dir results/tiny_gpt2/plots
```

`--summary` points to the CSV produced by the benchmark. `--output-dir` controls where plot PNGs are written. Use a new output directory, such as `results/tiny_gpt2`, when generating your own test results so they do not mix with previous runs.

5. Optional larger Mac run:

```bash
MODEL_NAME=Qwen/Qwen2.5-1.5B bash scripts/run_mac_full.sh
```

The full Mac script writes to `results/mac_full`. If it is too slow or runs out of memory, use a smaller model such as `Qwen/Qwen2.5-0.5B` or reduce `--seq-lens` / `--generated-tokens`.

## Quick Start On Google Colab

In Colab, choose `Runtime > Change runtime type > GPU`, then run:

```bash
!git clone <your-repo-url> kvcache
%cd kvcache
!pip install -r requirements-colab.txt
!nvidia-smi
!bash scripts/run_colab_smoke.sh
```

Full Colab run. By default this uses `Qwen/Qwen2.5-3B`, sequence lengths `128 256 512 1024`, 64 generated tokens, CUDA, and writes to `results/colab_full`:

```bash
!bash scripts/run_colab_full.sh
```

If Qwen2.5-3B does not fit in free Colab GPU memory, use Qwen2.5-1.5B:

```bash
!MODEL_NAME=Qwen/Qwen2.5-1.5B bash scripts/run_colab_full.sh
```

For a larger model on Colab, use a separate output directory and consider 4-bit model weights:

```bash
!MODEL_NAME=Qwen/Qwen2.5-7B \
  LOAD_IN_4BIT=true \
  OUTPUT_DIR=results/colab_qwen7b \
  SEQ_LENS="128 256 512" \
  GENERATED_TOKENS=32 \
  bash scripts/run_colab_full.sh
```

The Colab script accepts these environment variables:

- `MODEL_NAME`: Hugging Face causal LM id
- `OUTPUT_DIR`: where CSV, JSON, raw timing files, and quick plots are saved
- `SEQ_LENS`: space-separated prompt lengths
- `GENERATED_TOKENS`: number of decode tokens per run
- `ACCURACY_STEPS`: number of decode steps used for logit accuracy metrics
- `LOAD_IN_4BIT=true`: enables `--load-in-4bit` for CUDA runs with bitsandbytes

Equivalent direct Colab command:

```bash
python benchmarks/benchmark.py \
  --model-name Qwen/Qwen2.5-3B \
  --seq-lens 128 256 512 1024 \
  --generated-tokens 64 \
  --batch-size 1 \
  --block-size 16 \
  --run-adaptive-ablations true \
  --measure-accuracy true \
  --accuracy-steps 16 \
  --adaptive-monotonic true \
  --device cuda \
  --output-dir results/colab_qwen
```

A complete notebook version is available at `notebooks/colab_demo.ipynb`. It installs dependencies, checks `nvidia-smi`, runs smoke/full benchmarks, plots results, and zips the results for download.

Plot a specific Colab run by pointing `--summary` at that run's `summary_runs.csv` and writing plots into the same output directory:

```bash
!python benchmarks/plot_results.py \
  --summary results/colab_qwen7b/summary_runs.csv \
  --output-dir results/colab_qwen7b/plots
```

## Quick Start On Windows

Use PowerShell from the repo root.

CPU setup:

```powershell
cd C:\path\to\kvcache
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-windows.txt
python scripts\smoke_test.py
.\scripts\run_windows_smoke.ps1
```

Windows NVIDIA GPU setup depends on your CUDA version. Install the matching PyTorch CUDA wheel first. For example, with CUDA 12.1:

```powershell
cd C:\path\to\kvcache
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-windows.txt
.\scripts\run_windows_smoke.ps1
```

Windows full benchmark:

```powershell
.\scripts\run_windows_full.ps1
```

Equivalent direct Windows PowerShell command:

```powershell
python benchmarks/benchmark.py `
  --model-name Qwen/Qwen2.5-1.5B `
  --seq-lens 128 256 512 `
  --generated-tokens 32 `
  --batch-size 1 `
  --block-size 16 `
  --run-adaptive-ablations true `
  --measure-accuracy true `
  --accuracy-steps 16 `
  --adaptive-monotonic true `
  --device auto `
  --output-dir results/windows_qwen
```

Use PowerShell backticks for multiline commands. On Windows, `--device auto` uses CUDA if available and CPU otherwise.

## Running All Benchmarks

This runs baseline, INT8, blocked INT8, and the adaptive ablation modes:

```bash
python benchmarks/benchmark.py \
  --model-name Qwen/Qwen2.5-1.5B \
  --seq-lens 128 256 512 \
  --generated-tokens 32 \
  --batch-size 1 \
  --block-size 16 \
  --run-adaptive-ablations true \
  --measure-accuracy true \
  --accuracy-steps 16 \
  --adaptive-monotonic true \
  --device auto \
  --output-dir results/full_run
```

## Running Adaptive Ablations

`--run-adaptive-ablations true` expands to:

- `baseline`
- `int8`
- `blocked_int8`
- `adaptive_recency_k1`
- `adaptive_recency_k16`
- `adaptive_recency_k32`
- `adaptive_attention_k16`
- `adaptive_hybrid_k16`

Attention and hybrid modes enable `output_attentions=True`, so they are slower. Recency mode is the fast adaptive policy.

## Measuring Accuracy Drop

Accuracy metrics are enabled with:

```bash
--measure-accuracy true --accuracy-steps 16
```

The benchmark compares each mode against the baseline FP16 KV-cache run for the same prompt. It records cosine similarity, MAE, MSE, KL divergence, next-token agreement, generated-token overlap, and edit distance in the summary/JSON outputs.

## Plotting Results

After a benchmark finishes:

```bash
python benchmarks/plot_results.py \
  --summary results/summary_runs.csv \
  --output-dir results/plots
```

For platform-specific output directories, change the summary path, for example:

```bash
python benchmarks/plot_results.py \
  --summary results/colab_full/summary_runs.csv \
  --output-dir results/colab_full/plots
```

Generated plots include tokens/sec, latency/token, KV memory, compression ratio, cosine similarity, KL divergence, and next-token agreement versus sequence length.

## Where Results Are Saved

Each benchmark output directory contains:

- `summary_runs.csv`: append-friendly summary table for all modes and sequence lengths
- `benchmark_results.csv`: legacy per-run benchmark table
- `benchmark_results.json`: legacy JSON table
- `runs/{run_id}.json`: full config, device info, generated text, timing breakdown, and metrics
- `raw/{run_id}_timing.csv`: per-step latency and adaptive timing breakdown
- `raw/{run_id}_accuracy.csv`: per-step logit/argmax accuracy metrics
- `benchmark_tokens_per_sec.png`: quick throughput plot
- `plots/`: plots generated by `benchmarks/plot_results.py`

## Troubleshooting

macOS:

- If Mac runs out of memory, use `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-0.5B`, `distilgpt2`, or reduce `--seq-lens` / `--generated-tokens`.
- If MPS fails for an operation, rerun with `--device cpu`. It will be slower but more compatible.
- Do not install `bitsandbytes` on Mac unless you know your environment supports it; it is not required for these Mac scripts.

Colab:

- If Colab runs out of memory, use `MODEL_NAME=Qwen/Qwen2.5-1.5B bash scripts/run_colab_full.sh`, reduce sequence lengths, or reduce generated tokens.
- If Hugging Face rate limits downloads, run `huggingface-cli login` with your token.
- `bitsandbytes` is optional and mainly useful for large 4-bit weight experiments such as Mistral-7B.

Windows:

- If CUDA is not detected, install the PyTorch CUDA wheel matching your NVIDIA driver/CUDA runtime. Check with `python -c "import torch; print(torch.cuda.is_available())"`.
- `bitsandbytes` support on Windows can be limited, so keep it optional.
- If memory is insufficient, use `Qwen/Qwen2.5-0.5B` or `distilgpt2`.
- Use PowerShell backticks, not Bash backslashes, for multiline commands.

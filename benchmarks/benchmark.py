from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kv_cache_quant.adaptive_kv_policy import AdaptiveKVPolicy
from kv_cache_quant.baseline_decode import decode
from kv_cache_quant.similarity import compare_outputs
from kv_cache_quant.utils import default_dtype_for_device, select_device, set_seed

SUMMARY_COLUMNS = [
    "timestamp",
    "run_id",
    "model_name",
    "device",
    "dtype",
    "mode",
    "sequence_length",
    "generated_tokens",
    "batch_size",
    "tokens_per_sec",
    "latency_per_token_ms",
    "kv_memory_mb",
    "compression_ratio",
    "accuracy_cosine",
    "accuracy_mae",
    "accuracy_mse",
    "kl_divergence",
    "next_token_agreement",
    "generated_text_overlap",
    "adaptive_policy",
    "adaptive_update_interval",
    "adaptive_monotonic",
    "block_size",
    "notes",
]


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_modes(values):
    if values is None:
        return ["baseline", "int8", "blocked_int8", "adaptive"]
    if isinstance(values, str):
        values = [values]
    modes = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                modes.append(item)
    return modes or ["baseline", "int8", "blocked_int8", "adaptive"]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark KV-cache quantization decoding modes.")
    parser.add_argument("--model", default=None, help="Backward-compatible alias for --model-name.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B", help="Hugging Face causal LM id.")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=None, help="Backward-compatible list of sequence lengths.")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Backward-compatible alias for --generated-tokens.")
    parser.add_argument("--generated-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--prompt", default="The history of efficient neural network inference is")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--modes", nargs="+", default=None, help="Backward-compatible space/comma list of modes.")
    parser.add_argument("--mode", default=None, help="Comma-separated modes, e.g. baseline,int8,blocked_int8,adaptive.")
    parser.add_argument("--adaptive-policy", choices=["recency", "attention", "hybrid"], default="recency")
    parser.add_argument("--importance-mode", choices=["recency", "attention", "hybrid"], default=None, help="Backward-compatible alias for --adaptive-policy.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Recency weight for hybrid importance.")
    parser.add_argument("--beta", type=float, default=0.5, help="Attention weight for hybrid importance.")
    parser.add_argument("--fp16-ratio", type=float, default=0.2, help="Top block fraction kept in FP16 for adaptive mode.")
    parser.add_argument("--int8-ratio", type=float, default=0.5, help="Middle block fraction stored as INT8 for adaptive mode; remainder is INT4.")
    parser.add_argument("--adaptive-update-interval", type=int, default=16)
    parser.add_argument("--adaptive-monotonic", type=str_to_bool, default=True)
    parser.add_argument("--save-results", type=str_to_bool, default=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--measure-accuracy", type=str_to_bool, default=True)
    parser.add_argument("--accuracy-steps", type=int, default=16)
    parser.add_argument("--fixed-seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="eager", help="Use eager attention so output_attentions works for attention/hybrid adaptive policies.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Optional CUDA-only 4-bit model weights via bitsandbytes, useful for Mistral-7B.")
    parser.add_argument("--notes", default="")
    parser.add_argument("--run-adaptive-ablations", type=str_to_bool, default=False)
    return parser.parse_args()


def model_context_limit(model, tokenizer):
    candidates = [
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "n_positions", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    candidates = [int(x) for x in candidates if isinstance(x, int) and x < 1_000_000]
    return min(candidates) if candidates else None


def make_prompt(tokenizer, text: str, seq_len: int, batch_size: int):
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded["input_ids"]
    if ids.shape[1] == 0:
        raise ValueError("Prompt produced no tokens.")
    repeats = (seq_len + ids.shape[1] - 1) // ids.shape[1]
    ids = ids.repeat(1, repeats)[:, :seq_len]
    ids = ids.repeat(batch_size, 1)
    mask = ids.new_ones(ids.shape)
    return ids, mask


def load_causal_lm(model_id, dtype, device, attn_implementation="eager", load_in_4bit=False):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    if load_in_4bit:
        if device.type != "cuda":
            raise SystemExit("--load-in-4bit requires a CUDA GPU with bitsandbytes installed.")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit("Install bitsandbytes-compatible transformers support for --load-in-4bit.") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype if device.type != "cpu" else torch.float32,
            **kwargs,
        )
    except TypeError:
        kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype if device.type != "cpu" else torch.float32,
            **kwargs,
        )


def expand_mode(mode_name, default_policy, default_interval):
    if mode_name.startswith("adaptive_"):
        parts = mode_name.split("_")
        policy = parts[1] if len(parts) > 1 else default_policy
        interval = default_interval
        for part in parts[2:]:
            if part.startswith("k") and part[1:].isdigit():
                interval = int(part[1:])
        return "adaptive", policy, interval, mode_name
    return mode_name, default_policy, default_interval, mode_name


def build_mode_list(args):
    if args.run_adaptive_ablations:
        return [
            "baseline",
            "int8",
            "blocked_int8",
            "adaptive_recency_k1",
            "adaptive_recency_k16",
            "adaptive_recency_k32",
            "adaptive_attention_k16",
            "adaptive_hybrid_k16",
        ]
    if args.mode:
        return parse_modes(args.mode)
    return parse_modes(args.modes)


def append_csv(path: Path, fieldnames, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def hardware_info(torch, device):
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info


def main():
    args = parse_args()
    if args.model:
        args.model_name = args.model
    if args.max_new_tokens is not None:
        args.generated_tokens = args.max_new_tokens
    if args.importance_mode:
        args.adaptive_policy = args.importance_mode
    seq_lens = args.seq_lens or ([args.sequence_length] if args.sequence_length else [128, 256, 512])
    modes = build_mode_list(args)
    if args.measure_accuracy and "baseline" not in modes:
        modes = ["baseline"] + modes

    set_seed(args.fixed_seed)

    try:
        import pandas as pd
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install dependencies first with `pip install -r requirements.txt`.") from exc

    device = select_device(None if args.device == "auto" else args.device)
    dtype = default_dtype_for_device(device)
    if any("adaptive_attention" in mode or "adaptive_hybrid" in mode for mode in modes) or args.adaptive_policy in {"attention", "hybrid"}:
        print("Warning: attention/hybrid adaptive policies require output_attentions=True and can be much slower.")
    print(f"Loading {args.model_name} on {device} with dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_causal_lm(
        args.model_name,
        dtype=dtype,
        device=device,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
    )
    if not args.load_in_4bit:
        model.to(device)
    model.eval()
    context_limit = model_context_limit(model, tokenizer)

    output_dir = Path(args.output_dir)
    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    legacy_rows = []
    raw_timing_rows = []
    raw_accuracy_rows = []
    run_details = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config": vars(args) | {"sequence_lengths": seq_lens, "modes": modes},
        "hardware": hardware_info(torch, device),
        "results": [],
    }

    for seq_len in seq_lens:
        effective_seq_len = seq_len
        if context_limit is not None and seq_len + args.generated_tokens > context_limit:
            effective_seq_len = max(1, context_limit - args.generated_tokens)
            print(
                f"Requested sequence_length={seq_len} exceeds context limit {context_limit} "
                f"with generated_tokens={args.generated_tokens}; using {effective_seq_len}."
            )
        input_ids, attention_mask = make_prompt(tokenizer, args.prompt, effective_seq_len, args.batch_size)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        print(f"\nSequence length {effective_seq_len}")

        mode_results = {}
        baseline_result = None
        for requested_mode in modes:
            actual_mode, policy_name, update_interval, display_mode = expand_mode(
                requested_mode,
                args.adaptive_policy,
                args.adaptive_update_interval,
            )
            if actual_mode == "baseline" and baseline_result is not None:
                result = baseline_result
            else:
                policy = AdaptiveKVPolicy(
                    block_size=args.block_size,
                    fp16_ratio=args.fp16_ratio,
                    int8_ratio=args.int8_ratio,
                    importance_mode=policy_name,
                    alpha=args.alpha,
                    beta=args.beta,
                    update_interval=update_interval,
                    monotonic=args.adaptive_monotonic,
                )
                set_seed(args.fixed_seed)
                result = decode(
                    model,
                    tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.generated_tokens,
                    mode=actual_mode,
                    block_size=args.block_size,
                    adaptive_policy=policy,
                )
                if actual_mode == "baseline":
                    baseline_result = result
            mode_results[display_mode] = (actual_mode, policy_name, update_interval, result)

        baseline = baseline_result or next(iter(mode_results.values()))[3]
        for display_mode, (actual_mode, policy_name, update_interval, result) in mode_results.items():
            accuracy_logits = result.step_logits
            baseline_logits = baseline.step_logits
            if args.measure_accuracy and args.accuracy_steps > 0:
                limit = min(args.accuracy_steps, len(baseline_logits), len(accuracy_logits))
                baseline_logits = baseline_logits[:limit]
                accuracy_logits = accuracy_logits[:limit]
            sim = compare_outputs(baseline.generated_ids, result.generated_ids, baseline_logits, accuracy_logits)
            kv_mb = result.kv_cache_bytes / (1024**2)
            row = {
                "timestamp": timestamp,
                "run_id": run_id,
                "model_name": args.model_name,
                "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
                "mode": display_mode,
                "sequence_length": effective_seq_len,
                "generated_tokens": args.generated_tokens,
                "batch_size": args.batch_size,
                "tokens_per_sec": result.tokens_per_sec,
                "latency_per_token_ms": result.latency_ms_per_token,
                "kv_memory_mb": kv_mb,
                "compression_ratio": result.compression_ratio,
                "accuracy_cosine": sim.cosine_similarity,
                "accuracy_mae": sim.mean_abs_logit_error,
                "accuracy_mse": sim.mean_squared_error,
                "kl_divergence": sim.kl_divergence,
                "next_token_agreement": sim.next_token_agreement,
                "generated_text_overlap": sim.generated_text_overlap,
                "adaptive_policy": policy_name if actual_mode == "adaptive" else "",
                "adaptive_update_interval": update_interval if actual_mode == "adaptive" else "",
                "adaptive_monotonic": args.adaptive_monotonic if actual_mode == "adaptive" else "",
                "block_size": args.block_size,
                "notes": args.notes,
            }
            summary_rows.append(row)
            legacy_rows.append(
                {
                    "model": args.model_name,
                    "device": str(device),
                    "dtype": str(dtype).replace("torch.", ""),
                    "mode": display_mode,
                    "requested_seq_len": seq_len,
                    "seq_len": effective_seq_len,
                    "batch_size": args.batch_size,
                    "max_new_tokens": args.generated_tokens,
                    "block_size": args.block_size,
                    "seconds": result.seconds,
                    "tokens_per_sec": result.tokens_per_sec,
                    "latency_ms_per_token": result.latency_ms_per_token,
                    "kv_cache_mb": kv_mb,
                    "compression_ratio": result.compression_ratio,
                    **sim.as_dict(),
                    "output_text": result.text,
                }
            )
            for timing in result.step_timings:
                raw_timing_rows.append(
                    {
                        "step": timing["step"],
                        "mode": display_mode,
                        "sequence_length": effective_seq_len,
                        "latency_ms": timing["latency_ms"],
                        "tokens_per_sec": timing["tokens_per_sec"],
                        "kv_memory_mb": timing["kv_memory_mb"],
                        "model_forward_ms": timing.get("model_forward_ms", 0.0),
                        "importance_scoring_ms": timing.get("importance_scoring_ms", 0.0),
                        "quantization_ms": timing.get("quantization_ms", 0.0),
                        "dequantization_ms": timing.get("dequantization_ms", 0.0),
                        "precision_assignment_ms": timing.get("precision_assignment_ms", 0.0),
                        "tensor_copy_relayout_ms": timing.get("tensor_copy_relayout_ms", 0.0),
                    }
                )
            for acc in sim.per_step:
                raw_accuracy_rows.append({"mode": display_mode, **acc})
            run_details["results"].append(
                {
                    **row,
                    "actual_mode": actual_mode,
                    "requested_sequence_length": seq_len,
                    "generated_baseline_text": baseline.text,
                    "generated_test_mode_text": result.text,
                    "timing_breakdown": result.timing_breakdown,
                    "precision_counts": result.precision_counts,
                    "accuracy_metrics": sim.as_dict(),
                }
            )
            print(
                f"{display_mode:24s} {result.tokens_per_sec:8.2f} tok/s "
                f"{result.latency_ms_per_token:8.2f} ms/token {kv_mb:8.3f} MB "
                f"comp={result.compression_ratio:5.2f}x cos={sim.cosine_similarity:.4f} kl={sim.kl_divergence:.4g}"
            )

    if args.save_results:
        append_csv(output_dir / "summary_runs.csv", SUMMARY_COLUMNS, summary_rows)
        write_csv(
            output_dir / "raw" / f"{run_id}_timing.csv",
            [
                "step",
                "mode",
                "sequence_length",
                "latency_ms",
                "tokens_per_sec",
                "kv_memory_mb",
                "model_forward_ms",
                "importance_scoring_ms",
                "quantization_ms",
                "dequantization_ms",
                "precision_assignment_ms",
                "tensor_copy_relayout_ms",
            ],
            raw_timing_rows,
        )
        write_csv(
            output_dir / "raw" / f"{run_id}_accuracy.csv",
            [
                "step",
                "mode",
                "cosine_similarity",
                "mae",
                "mse",
                "kl_divergence",
                "baseline_argmax_token",
                "test_argmax_token",
                "argmax_match",
            ],
            raw_accuracy_rows,
        )
        (output_dir / "runs" / f"{run_id}.json").write_text(json.dumps(run_details, indent=2), encoding="utf-8")

    df = pd.DataFrame(legacy_rows)
    csv_path = output_dir / "benchmark_results.csv"
    json_path = output_dir / "benchmark_results.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(legacy_rows, indent=2), encoding="utf-8")
    print(f"\nSaved {csv_path}")
    print(f"Saved {json_path}")
    if args.save_results:
        print(f"Saved {output_dir / 'summary_runs.csv'}")
        print(f"Saved {output_dir / 'runs' / (run_id + '.json')}")
        print(f"Saved {output_dir / 'raw' / (run_id + '_timing.csv')}")
        print(f"Saved {output_dir / 'raw' / (run_id + '_accuracy.csv')}")

    try:
        import matplotlib.pyplot as plt

        pivot = df.pivot_table(index="seq_len", columns="mode", values="tokens_per_sec", aggfunc="mean")
        ax = pivot.plot(marker="o", title="Decode Throughput")
        ax.set_ylabel("tokens/sec")
        ax.set_xlabel("sequence length")
        fig = ax.get_figure()
        plot_path = output_dir / "benchmark_tokens_per_sec.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        print(f"Saved {plot_path}")
    except Exception as exc:
        print(f"Plot skipped: {exc}")


if __name__ == "__main__":
    main()

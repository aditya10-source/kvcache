from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot saved KV-cache benchmark summaries.")
    parser.add_argument("--summary", default="results/summary_runs.csv")
    parser.add_argument("--output-dir", default="results/plots")
    return parser.parse_args()


def plot_metric(df, metric, ylabel, output_dir):
    import matplotlib.pyplot as plt

    if metric not in df.columns:
        print(f"Skipping {metric}: missing column")
        return
    clean = df.dropna(subset=[metric, "sequence_length", "mode"])
    if clean.empty:
        print(f"Skipping {metric}: no data")
        return
    pivot = clean.pivot_table(index="sequence_length", columns="mode", values=metric, aggfunc="mean")
    ax = pivot.plot(marker="o", title=ylabel)
    ax.set_xlabel("sequence length")
    ax.set_ylabel(ylabel)
    fig = ax.get_figure()
    fig.tight_layout()
    path = output_dir / f"{metric}_vs_sequence_length.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    args = parse_args()
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Install pandas/matplotlib with `pip install -r requirements.txt`.") from exc

    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_path)
    metrics = [
        ("tokens_per_sec", "tokens/sec"),
        ("latency_per_token_ms", "latency per token (ms)"),
        ("kv_memory_mb", "KV memory (MB)"),
        ("compression_ratio", "compression ratio"),
        ("accuracy_cosine", "accuracy cosine similarity"),
        ("kl_divergence", "KL divergence"),
        ("next_token_agreement", "next-token agreement"),
    ]
    for metric, ylabel in metrics:
        plot_metric(df, metric, ylabel, output_dir)


if __name__ == "__main__":
    main()

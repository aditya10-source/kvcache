from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "smoke_benchmark"
RUN_ID = "smoke_benchmark"


def main():
    cmd = [
        sys.executable,
        str(ROOT / "benchmarks" / "benchmark.py"),
        "--model-name",
        "sshleifer/tiny-gpt2",
        "--sequence-length",
        "64",
        "--generated-tokens",
        "8",
        "--mode",
        "baseline,int8,blocked_int8,adaptive_recency_k16",
        "--run-id",
        RUN_ID,
        "--output-dir",
        str(OUT),
        "--save-results",
        "true",
        "--measure-accuracy",
        "true",
        "--accuracy-steps",
        "8",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)

    required = [
        OUT / "summary_runs.csv",
        OUT / "runs" / f"{RUN_ID}.json",
        OUT / "raw" / f"{RUN_ID}_timing.csv",
        OUT / "raw" / f"{RUN_ID}_accuracy.csv",
    ]
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"Missing expected result file: {path}")

    with (OUT / "summary_runs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("summary_runs.csv has no rows")
    for row in rows:
        for col in ["accuracy_cosine", "accuracy_mae", "accuracy_mse", "kl_divergence", "next_token_agreement"]:
            value = float(row[col])
            if math.isnan(value) or math.isinf(value):
                raise SystemExit(f"Non-finite {col} for mode {row['mode']}: {value}")

    print("smoke_benchmark passed")
    for path in required:
        print(path)


if __name__ == "__main__":
    main()

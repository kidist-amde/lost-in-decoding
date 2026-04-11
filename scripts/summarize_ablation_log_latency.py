#!/usr/bin/env python3
"""Summarize estimated per-iteration latency from tqdm progress logs.

This parser reads .err/.out logs and extracts:
- throughput matches like "12.34it/s"
- duration matches like "0.81s/it"

It converts both forms to estimated milliseconds per iteration and prints
per-file summary statistics.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from statistics import mean, median


ITS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*it/s")
SIT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*s/it")


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    idx = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    w = idx - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def parse_ms_values(text: str) -> list[float]:
    ms_values: list[float] = []
    for m in ITS_RE.finditer(text):
        itps = float(m.group(1))
        if itps > 0:
            ms_values.append(1000.0 / itps)
    for m in SIT_RE.finditer(text):
        s_per_it = float(m.group(1))
        if s_per_it > 0:
            ms_values.append(1000.0 * s_per_it)
    return ms_values


def summarize_file(path: Path) -> dict[str, float | int | str]:
    text = path.read_text(errors="ignore")
    ms_values = parse_ms_values(text)
    if not ms_values:
        return {
            "file": str(path),
            "count": 0,
        }

    ms_sorted = sorted(ms_values)
    return {
        "file": str(path),
        "count": len(ms_sorted),
        "mean_ms": mean(ms_sorted),
        "p50_ms": median(ms_sorted),
        "p95_ms": percentile(ms_sorted, 95),
        "min_ms": ms_sorted[0],
        "max_ms": ms_sorted[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize estimated latency from tqdm speed fields in logs."
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        default="experiments/Ablations",
        help="Directory containing .err/.out logs (default: experiments/Ablations)",
    )
    parser.add_argument(
        "--glob",
        default="*.err",
        help="Glob pattern within log_dir (default: *.err)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    paths = sorted(log_dir.glob(args.glob))
    if not paths:
        print(f"No files matched: {log_dir}/{args.glob}")
        return

    print("file,count,mean_ms,p50_ms,p95_ms,min_ms,max_ms")
    for p in paths:
        row = summarize_file(p)
        if row["count"] == 0:
            print(f"{row['file']},0,,,,,")
            continue
        print(
            f"{row['file']},{row['count']},"
            f"{row['mean_ms']:.3f},{row['p50_ms']:.3f},{row['p95_ms']:.3f},"
            f"{row['min_ms']:.3f},{row['max_ms']:.3f}"
        )


if __name__ == "__main__":
    main()

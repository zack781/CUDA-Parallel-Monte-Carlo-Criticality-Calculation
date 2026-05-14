#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def percentile(values, q):
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def main():
    parser = argparse.ArgumentParser(
        description="Plot the distribution of move_kernel calls per source history."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="history_move_counts.csv",
        help="CSV produced by a PROFILE_HISTORY_LENGTHS=1 transport_sim run.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults next to the CSV.",
    )
    args = parser.parse_args()

    counts = []
    with open(args.csv_path, newline="") as f:
        for row in csv.DictReader(f):
            counts.append(int(row["move_kernel_calls"]))

    if not counts:
        raise SystemExit("No history move counts found.")

    counts.sort()
    mean = sum(counts) / len(counts)
    p50 = percentile(counts, 0.50)
    p90 = percentile(counts, 0.90)
    p99 = percentile(counts, 0.99)
    max_count = counts[-1]

    print(f"histories: {len(counts)}")
    print(f"mean move calls/history: {mean:.2f}")
    print(f"p50: {p50:.0f}")
    print(f"p90: {p90:.0f}")
    print(f"p99: {p99:.0f}")
    print(f"max: {max_count}")

    bins = range(0, max_count + 2)
    plt.figure(figsize=(7, 4.5))
    plt.hist(counts, bins=bins, edgecolor="black", linewidth=0.3)
    plt.yscale("log")
    plt.xlabel("move_kernel calls per history")
    plt.ylabel("Number of histories")
    plt.title("History Length Distribution")
    plt.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output = Path(args.output) if args.output else Path(args.csv_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

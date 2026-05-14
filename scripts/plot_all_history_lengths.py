#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def neutron_count_from_name(path):
    match = re.search(r"_N(\d+)_", path.name)
    if match is None:
        raise ValueError(f"Could not parse neutron count from {path}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(
        description="Plot all history-length CSVs and summarize mean calls vs neutron count."
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        default="results/history_lengths",
        help="Directory containing history_moves_N*_G*_B*.csv files.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    csv_paths = sorted(result_dir.glob("history_moves_N*_G*_B*.csv"), key=neutron_count_from_name)
    if not csv_paths:
        raise SystemExit(f"No history move CSVs found in {result_dir}")

    summaries = []
    for csv_path in csv_paths:
        counts = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                counts.append(int(row["move_kernel_calls"]))

        if not counts:
            continue

        neutrons = neutron_count_from_name(csv_path)
        mean_calls = sum(counts) / len(counts)
        max_calls = max(counts)
        summaries.append((neutrons, mean_calls, max_calls))

        output = csv_path.with_suffix(".png")
        subprocess.run(
            [
                sys.executable,
                "scripts/plot_history_move_counts.py",
                str(csv_path),
                "--output",
                str(output),
            ],
            check=True,
        )

    summaries.sort()
    neutrons = [row[0] for row in summaries]
    mean_calls = [row[1] for row in summaries]
    max_calls = [row[2] for row in summaries]

    plt.figure(figsize=(7, 4.5))
    plt.plot(neutrons, mean_calls, marker="o", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Neutrons per generation")
    plt.ylabel("Average move_kernel calls per history")
    plt.title("Average History Length vs Neutron Count")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output = result_dir / "average_move_calls_vs_neutrons.png"
    plt.savefig(output, dpi=200)
    print(f"Wrote {output}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(neutrons, max_calls, marker="o", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Neutrons per generation")
    plt.ylabel("Maximum move_kernel calls for one history")
    plt.title("Maximum History Length vs Neutron Count")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output = result_dir / "max_move_calls_vs_neutrons.png"
    plt.savefig(output, dpi=200)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

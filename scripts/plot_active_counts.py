#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def neutron_count_from_name(path):
    match = re.search(r"_N(\d+)_", path.name)
    if match is None:
        raise ValueError(f"Could not parse neutron count from {path}")
    return int(match.group(1))


def load_average_trace(path):
    by_iteration = defaultdict(list)
    max_iterations = []
    by_generation_max = defaultdict(int)

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            generation = int(row["generation"])
            iteration = int(row["iteration"])
            active_fraction = float(row["active_fraction"])
            by_iteration[iteration].append(active_fraction)
            by_generation_max[generation] = max(by_generation_max[generation], iteration)

    for iteration in by_generation_max.values():
        max_iterations.append(iteration)

    iterations = sorted(by_iteration)
    averaged = [
        sum(by_iteration[iteration]) / len(by_iteration[iteration])
        for iteration in iterations
    ]
    return iterations, averaged, max_iterations


def main():
    parser = argparse.ArgumentParser(
        description="Plot active particle count traces from PROFILE_ACTIVE_COUNTS CSVs."
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        default="results/active_counts",
        help="Directory containing active_counts_N*_G*_B*.csv files.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    csv_paths = sorted(result_dir.glob("active_counts_N*_G*_B*.csv"), key=neutron_count_from_name)
    if not csv_paths:
        raise SystemExit(f"No active-count CSVs found in {result_dir}")

    plt.figure(figsize=(8, 5))
    max_iteration_summary = []
    for csv_path in csv_paths:
        neutrons = neutron_count_from_name(csv_path)
        iterations, active_fractions, max_iterations = load_average_trace(csv_path)
        positive = [(i, f) for i, f in zip(iterations, active_fractions) if f > 0.0]
        if positive:
            x, y = zip(*positive)
            plt.plot(x, y, linewidth=1.8, label=f"N={neutrons}")
        if max_iterations:
            max_iteration_summary.append(
                (neutrons, sum(max_iterations) / len(max_iterations), max(max_iterations))
            )

    plt.yscale("log")
    plt.xlabel("Transport iteration")
    plt.ylabel("Mean active fraction")
    plt.title("Active Histories Remaining vs Transport Iteration")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    output = result_dir / "active_fraction_vs_iteration.png"
    plt.savefig(output, dpi=200)
    print(f"Wrote {output}")

    max_iteration_summary.sort()
    neutrons = [row[0] for row in max_iteration_summary]
    mean_last_iterations = [row[1] for row in max_iteration_summary]
    max_last_iterations = [row[2] for row in max_iteration_summary]

    plt.figure(figsize=(7, 4.5))
    plt.plot(neutrons, mean_last_iterations, marker="o", linewidth=2, label="Mean final iteration")
    plt.plot(neutrons, max_last_iterations, marker="s", linewidth=2, label="Max final iteration")
    plt.xscale("log")
    plt.xlabel("Neutrons per generation")
    plt.ylabel("Final transport iteration")
    plt.title("Transport Tail Length vs Neutron Count")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    output = result_dir / "tail_iterations_vs_neutrons.png"
    plt.savefig(output, dpi=200)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

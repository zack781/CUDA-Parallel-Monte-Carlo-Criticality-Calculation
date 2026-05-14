#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot scaling metrics vs neutron count for one batch size."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="results/perf/summary.csv",
        help="Path to performance summary CSV.",
    )
    parser.add_argument(
        "--output",
        default="results/perf/batch_size_1_wall_time.png",
        help="Output wall-time PNG path.",
    )
    parser.add_argument(
        "--batch-size",
        default="1",
        help="Batch size to plot.",
    )
    args = parser.parse_args()

    rows = []
    with open(args.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["batch_size"] != args.batch_size:
                continue
            if row["wall_seconds"] == "NA":
                continue
            rows.append(
                (
                    int(row["neutrons"]),
                    float(row["wall_seconds"]),
                    float(row["gpu_seconds"]) if row.get("gpu_seconds", "NA") != "NA" else None,
                    int(row["completed_generations"]),
                    int(row["interactions"]),
                )
            )

    if not rows:
        raise SystemExit("No batch_size=1 rows with valid wall_seconds found.")

    rows.sort()
    neutrons = [r[0] for r in rows]
    wall_seconds = [r[1] for r in rows]
    x_limits = (min(neutrons), max(neutrons))
    gpu_rows = [r for r in rows if r[2] is not None]
    if not gpu_rows:
        raise SystemExit("No rows with valid gpu_seconds found for throughput plots.")

    gpu_neutrons = [r[0] for r in gpu_rows]
    histories_per_second = [r[0] * r[3] / r[2] for r in gpu_rows]
    interactions_per_second = [r[4] / r[2] for r in gpu_rows]
    time_values = wall_seconds + [r[2] for r in gpu_rows]
    time_limits = (min(time_values) * 0.8, max(time_values) * 1.25)

    plt.figure(figsize=(7, 4.5))
    plt.plot(neutrons, wall_seconds, marker="o", linewidth=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(*x_limits)
    plt.ylim(*time_limits)
    plt.xlabel("Neutrons per generation")
    plt.ylabel("Wall time (seconds)")
    plt.title(f"Wall Time Scaling, Batch Size {args.batch_size}")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Wrote {output}")

    if gpu_rows:
        gpu_output = output.with_name(
            output.stem.replace("wall_time", "gpu_timed_section") + output.suffix
        )
        plt.figure(figsize=(7, 4.5))
        plt.plot([r[0] for r in gpu_rows], [r[2] for r in gpu_rows], marker="o", linewidth=2)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(*x_limits)
        plt.ylim(*time_limits)
        plt.xlabel("Neutrons per generation")
        plt.ylabel("GPU timed section (seconds)")
        plt.title(f"GPU Timed Section Scaling, Batch Size {args.batch_size}")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(gpu_output, dpi=200)
        print(f"Wrote {gpu_output}")

    histories_output = output.with_name(
        output.stem.replace("wall_time", "histories_per_second") + output.suffix
    )
    plt.figure(figsize=(7, 4.5))
    plt.plot(gpu_neutrons, histories_per_second, marker="o", linewidth=2)
    plt.xscale("log")
    plt.xlim(*x_limits)
    plt.xlabel("Neutrons per generation")
    plt.ylabel("Histories per GPU second")
    plt.title(f"GPU History Throughput, Batch Size {args.batch_size}")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(histories_output, dpi=200)
    print(f"Wrote {histories_output}")

    interactions_output = output.with_name(
        output.stem.replace("wall_time", "interactions_per_second") + output.suffix
    )
    plt.figure(figsize=(7, 4.5))
    plt.plot(gpu_neutrons, interactions_per_second, marker="o", linewidth=2)
    plt.xscale("log")
    plt.xlim(*x_limits)
    plt.xlabel("Neutrons per generation")
    plt.ylabel("Interactions per GPU second")
    plt.title(f"GPU Interaction Throughput, Batch Size {args.batch_size}")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(interactions_output, dpi=200)
    print(f"Wrote {interactions_output}")


if __name__ == "__main__":
    main()

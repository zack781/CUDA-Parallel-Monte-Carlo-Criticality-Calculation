#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def neutron_count(path):
    match = re.search(r"_N(\d+)_", path.name)
    return int(match.group(1)) if match else -1


def page_title(path):
    if path.name == "average_move_calls_vs_neutrons.png":
        return "Average move_kernel Calls vs Neutron Count"
    if path.name == "max_move_calls_vs_neutrons.png":
        return "Maximum move_kernel Calls vs Neutron Count"

    match = re.search(r"_N(\d+)_G(\d+)_B(\d+)", path.name)
    if match:
        neutrons, generations, batch_size = match.groups()
        return f"History Length Distribution: N={neutrons}, G={generations}, B={batch_size}"

    return path.stem.replace("_", " ").title()


def main():
    parser = argparse.ArgumentParser(description="Combine labeled PNG plots into one PDF.")
    parser.add_argument(
        "result_dir",
        nargs="?",
        default="results/history_lengths",
        help="Directory containing PNG plots.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PDF path. Defaults to history_length_plots.pdf in result_dir.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output = Path(args.output) if args.output else result_dir / "history_length_plots.pdf"

    summary_plots = [
        result_dir / "average_move_calls_vs_neutrons.png",
        result_dir / "max_move_calls_vs_neutrons.png",
    ]
    history_plots = sorted(result_dir.glob("history_moves_N*_G*_B*.png"), key=neutron_count)
    png_paths = [path for path in summary_plots if path.exists()] + history_plots

    if not png_paths:
        raise SystemExit(f"No PNG plots found in {result_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for png_path in png_paths:
            image = mpimg.imread(png_path)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            fig.suptitle(page_title(png_path), fontsize=16, y=0.97)
            ax.imshow(image)
            ax.axis("off")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

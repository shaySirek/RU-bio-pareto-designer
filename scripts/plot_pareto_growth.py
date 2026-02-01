import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_pareto_metrics(data_file: Path):
    df = pd.read_csv(data_file)
    fig_file = data_file.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(
        df["position"],
        df["min_size_pareto_set"],
        df["max_size_pareto_set"],
        color="gray",
        alpha=0.2,
        label="Range (Min-Max)",
    )
    ax.fill_between(
        df["position"],
        df["q1_size_pareto_set"],
        df["q3_size_pareto_set"],
        color="blue",
        alpha=0.3,
        label="IQR ($Q_1$ to $Q_3$)",
    )
    ax.plot(
        df["position"],
        df["avg_size_pareto_set"],
        color="blue",
        lw=2,
        label="Mean ($\mu$)",
    )
    ax.plot(
        df["position"],
        df["median_size_pareto_set"],
        color="red",
        ls="--",
        label="Median ($\\tilde{x}$)",
    )

    ax.set_xlabel("Position ($i$)")
    ax.set_ylabel("Pareto Set Size")
    ax.legend()

    plt.tight_layout()
    fig.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="Path to the Pareto size CSV file")
    args = parser.parse_args()

    plot_pareto_metrics(args.file)


if __name__ == "__main__":
    main()

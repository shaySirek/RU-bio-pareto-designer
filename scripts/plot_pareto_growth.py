import argparse

import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def plot_pareto_metrics(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))

    plt.fill_between(
        df["position"],
        df["min_size_pareto_set"],
        df["max_size_pareto_set"],
        color="gray",
        alpha=0.2,
        label="Range (Min-Max)",
    )

    plt.fill_between(
        df["position"],
        df["q1_size_pareto_set"],
        df["q3_size_pareto_set"],
        color="blue",
        alpha=0.3,
        label="IQR ($Q_1$ to $Q_3$)",
    )

    plt.plot(
        df["position"],
        df["avg_size_pareto_set"],
        color="blue",
        lw=2,
        label="Mean ($\mu$)",
    )
    plt.plot(
        df["position"],
        df["median_size_pareto_set"],
        color="red",
        ls="--",
        label="Median ($\tilde{x}$)",
    )

    plt.xlabel("Position ($i$)")
    plt.ylabel("Pareto Set Size")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the Pareto size CSV file")
    args = parser.parse_args()

    df = load_data(args.file)
    plot_pareto_metrics(df)


if __name__ == "__main__":
    main()

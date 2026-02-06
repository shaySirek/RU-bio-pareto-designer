from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ParetoSet:
    def __init__(self, output_path: Path):
        self._data_file = output_path / "pareto_set_size.csv"
        self._growth_fig = output_path / "pareto_growth.png"
        self._pruning_fig = output_path / "pareto_pruning.png"
        self._first = True
        output_path.mkdir(parents=True, exist_ok=True)
        self._data_file.unlink(missing_ok=True)

    def report_row_size(self, i: int, pareto_set_sizes: list[int], n_pruned: int):
        mean_n_pruned = n_pruned / len(pareto_set_sizes)
        arr = np.array(pareto_set_sizes)
        mean_pareto = np.mean(arr)

        data = {
            "position": i,
            "mean_pareto_set_size": mean_pareto,
            "mean_pruned": mean_n_pruned,
            "mean_pareto_and_pruned": mean_pareto + mean_n_pruned,
            "min_pareto_set_size": np.min(arr),
            "q1_pareto_set_size": np.percentile(arr, 25),
            "median_pareto_set_size": np.median(arr),
            "q3_pareto_set_size": np.percentile(arr, 75),
            "max_pareto_set_size": np.max(arr),
        }

        with self._data_file.open("at") as f:
            if self._first:
                f.write(",".join(data.keys()) + "\n")
                self._first = False
            f.write(
                ",".join(
                    f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
                    for v in data.values()
                )
                + "\n"
            )

    def plot(self):
        df = pd.read_csv(self._data_file)
        self._plot_growth(df)
        self._plot_pruning(df)

    def _plot_growth(self, df):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(
            df["position"],
            df["min_pareto_set_size"],
            df["max_pareto_set_size"],
            color="gray",
            alpha=0.2,
            label="Range",
        )
        ax.fill_between(
            df["position"],
            df["q1_pareto_set_size"],
            df["q3_pareto_set_size"],
            color="blue",
            alpha=0.3,
            label="IQR",
        )
        ax.plot(
            df["position"],
            df["median_pareto_set_size"],
            color="red",
            ls="--",
            label="Median",
        )
        ax.set_xlabel("Position ($i$)")
        ax.set_ylabel("Pareto Set Size")
        ax.legend()
        plt.tight_layout()
        fig.savefig(self._growth_fig, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_pruning(self, df):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            df["position"],
            df["mean_pareto_set_size"],
            label="Pareto Set",
            color="blue",
            alpha=0.8,
        )
        ax.plot(
            df["position"],
            df["mean_pareto_and_pruned"],
            label="Union",
            ls=":",
            color="black",
        )
        ax.scatter(
            df["position"],
            df["mean_pruned"],
            label="Pruned Infinite Costs",
            color="orange",
            marker="x",
            s=20,
            alpha=0.6,
        )

        ax.set_xlabel("Position ($i$)")
        ax.set_ylabel("Mean Size")
        ax.legend()
        plt.tight_layout()
        fig.savefig(self._pruning_fig, dpi=300, bbox_inches="tight")
        plt.close(fig)

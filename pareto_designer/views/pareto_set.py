from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ParetoSet:
    _DATA_FILENAME = "pareto_set_size.csv"
    _FIG_FILENAME = "pareto_growth.png"

    _PARETO_PREFIX = "pareto_set_size"
    _PRUNED_PREFIX = "pruned_size"

    def __init__(self, output_path: Path):
        self._data_file = output_path / self._DATA_FILENAME
        self._fig_file = output_path / self._FIG_FILENAME
        self._is_first_row = True
        output_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
        if arr.size == 0:
            return {f"{k}_{prefix}": 0.0 for k in ["min", "q1", "median", "q3", "max"]}

        return {
            f"min_{prefix}": np.min(arr),
            f"q1_{prefix}": np.percentile(arr, 25),
            f"median_{prefix}": np.median(arr),
            f"q3_{prefix}": np.percentile(arr, 75),
            f"max_{prefix}": np.max(arr),
        }

    def report_row_size(
        self,
        i: int,
        pareto_set_sizes: list[int],
        pruned_sizes: list[int],
    ):
        po_sizes_arr = np.array(pareto_set_sizes)
        pruned_sizes_arr = np.array(pruned_sizes)

        data = {"position": i}
        data.update(self._get_stats(po_sizes_arr, self._PARETO_PREFIX))
        data.update(self._get_stats(pruned_sizes_arr, self._PRUNED_PREFIX))

        mode = "wt" if self._is_first_row else "at"
        with self._data_file.open(mode) as f:
            if self._is_first_row:
                f.write(",".join(data.keys()) + "\n")
                self._is_first_row = False

            f.write(
                ",".join(
                    f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
                    for v in data.values()
                )
                + "\n"
            )

    def plot(self):
        df = pd.read_csv(self._data_file)
        fig, ax = plt.subplots(figsize=(15, 6))

        ax.fill_between(
            df["position"],
            df[f"min_{self._PARETO_PREFIX}"],
            df[f"max_{self._PARETO_PREFIX}"],
            color="gray",
            alpha=0.2,
            label="Range",
        )
        ax.fill_between(
            df["position"],
            df[f"q1_{self._PARETO_PREFIX}"],
            df[f"q3_{self._PARETO_PREFIX}"],
            color="blue",
            alpha=0.3,
            label="IQR",
        )
        ax.plot(
            df["position"],
            df[f"median_{self._PARETO_PREFIX}"],
            color="red",
            ls="--",
            lw=1.5,
            label="Median",
        )

        ax.set_xlabel("Position ($i$)")
        ax.set_ylabel("Size")
        ax.legend(loc="upper left")

        plt.tight_layout()
        fig.savefig(self._fig_file, dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":

    import sys

    ps = ParetoSet(Path(sys.argv[1]))
    ps.plot()

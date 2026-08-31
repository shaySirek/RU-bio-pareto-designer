import argparse
import math
from loguru import logger
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIXED_AVG = "fixed-avg"
FIXED_TOTAL = "fixed-total"


def weight_function(avg_cost: float, i: np.ndarray, alpha: float) -> np.ndarray:
    return (avg_cost + 1) ** (-alpha * math.log(i + 1))


def plot_selection_weight_scenario(
    scenario: str,
    alpha: float,
    max_i: int = 1000,
    ticks: int = 50,
):
    logger.info(
        f"Starting plot generation for scenario: '{scenario}' with alpha={alpha}"
    )

    i = np.arange(1, max_i + 1)
    fig, ax = plt.subplots(figsize=(12, 4))

    if scenario == FIXED_AVG:
        avg_costs = np.arange(0.0, 0.06, 0.01)
        for avg_cost in avg_costs.tolist():
            w = weight_function(avg_cost, i, alpha)
            ax.plot(i, w, label=rf"$\bar{{C}}$ = {avg_cost:.2f}", linewidth=2)
    elif scenario == FIXED_TOTAL:
        total_costs = np.arange(0.0, 6.0, 1.0)
        for c in total_costs.tolist():
            w = weight_function(c / i, i, alpha)
            ax.plot(i, w, label=f"c = {c:.1f}", linewidth=2)
    else:
        logger.error(f"Invalid scenario provided: {scenario}")
        raise ValueError(
            f"Invalid scenario. Choose either '{FIXED_AVG}' or '{FIXED_TOTAL}'."
        )

    ax.set_xlabel("$i$", fontsize=11)
    ax.set_ylabel(r"$W(\bar{C}, i)$", fontsize=11)

    ax.set_xlim(1, max_i)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xticks(np.arange(0, max_i + 1, ticks))
    ax.set_xticklabels(np.arange(0, max_i + 1, ticks), rotation=45)

    ax.legend(ncols=2)
    fig.tight_layout()

    filename = Path("plots") / f"selection_weight_{scenario}_alpha_{alpha}.png"
    fig.savefig(filename, dpi=300)
    logger.success(f"Successfully saved plot to {filename}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize selection weight asymptotic boundaries."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5.0,
        help="Selection pressure factor alpha (default: 5.0)",
    )

    args = parser.parse_args()

    plot_selection_weight_scenario(scenario=FIXED_AVG, alpha=args.alpha)
    plot_selection_weight_scenario(scenario=FIXED_TOTAL, alpha=args.alpha)

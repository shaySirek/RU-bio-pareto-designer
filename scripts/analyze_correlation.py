import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from scipy.stats import pearsonr
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="Results file")
    parser.add_argument(
        "-x", type=str, default="efficiency_gain_perc", help="Column name for x-axis"
    )
    parser.add_argument(
        "-y",
        type=str,
        default="motif_ic_[0-9]*",
        help="Regex for column names for y-axis",
    )
    return parser.parse_args()


def main(args):
    df = pd.read_csv(args.results)
    x = args.x
    y_columns = [
        col for col in df.columns if re.fullmatch(args.y.replace("*", ".*"), col)
    ]
    if not y_columns:
        raise ValueError(f"No columns matched y regex: {args.y}")

    fig, ax = plt.subplots(figsize=(15, 6))
    plt.subplots_adjust(right=0.75)  # Make room for checkboxes

    lines = []
    labels = []

    for y in y_columns:
        df_xy = df[[x, y]].dropna()
        x_vals = df_xy[x]
        y_vals = df_xy[y]
        corr, p = pearsonr(x_vals, y_vals)
        label = f"{y} [{df_xy.shape[0]}] (r={corr:.2f}, p={p:.1e})"
        (line,) = ax.plot(x_vals, y_vals, "o", label=label, alpha=0.7)
        lines.append(line)
        labels.append(label)

    x_label = x.replace("_", " ").title()
    ax.set_xlabel(x_label)
    ax.set_ylabel("Y values")
    ax.set_title(f"Correlation with {x_label}")

    rax = plt.axes([0.78, 0.25, 0.2, 0.5])
    check = CheckButtons(rax, labels, [True] * len(labels))

    def toggle(label):
        idx = labels.index(label)
        lines[idx].set_visible(not lines[idx].get_visible())
        plt.draw()

    check.on_clicked(toggle)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)

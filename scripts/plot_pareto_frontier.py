import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt


def render_pareto_front_png(results: list[dict], fig_file: Path):
    fig, ax = plt.subplots(figsize=(9, 4))

    colors = {"0": "#cbd5e0", "1": "#d8b4fe", "2-5": "#a855f7", "5+": "#6b21a8"}
    buckets = {key: [] for key in colors}
    for r in results:
        hits = len(r["motif_hits"])
        if hits < 2:
            if len(buckets[str(hits)]) == 0:
                print(
                    f"Solution no. {r['id']} is the first solution (lowest cost of {r['cost']:.3f}) with {hits} motif hits."
                )
            buckets[str(hits)].append(r)
        elif 2 <= hits <= 5:
            buckets["2-5"].append(r)
        else:
            buckets["5+"].append(r)

    groups = {key: (buckets[key], colors[key]) for key in colors}

    for label, (group_results, color) in groups.items():
        if not group_results:
            continue
        ax.scatter(
            [r["cost"] for r in group_results],
            [r["binding_score"] for r in group_results],
            label=label,
            c=color,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.8,
            s=60,
        )

    ax.set_xlabel("Functional Cost", fontsize=12)
    ax.set_ylabel("Binding Score", fontsize=12)
    ax.set_xlim(0, 500)
    ax.set_ylim(-21000, -14000)
    ax.legend(title="Motif Hits", loc="upper right")

    fig.savefig(
        fig_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    results_path = Path(sys.argv[1])
    data_file = results_path / "results_metadata.json"
    with data_file.open() as f:
        results: list[dict] = json.load(f)["results"]

    print(f"Found {len(results)} solutions in {data_file}")

    # remove exterme solutions
    highest_cost = float(results[-1]["cost"])
    while highest_cost > 5000:
        print(f"Remove solution with exterme cost of {highest_cost:.3f}")
        results.pop()
        highest_cost = float(results[-1]["cost"])

    print(f"Plot {len(results)} solutions")
    render_pareto_front_png(results, results_path / "pareto_optimal_solutions.png")


if __name__ == "__main__":
    main()

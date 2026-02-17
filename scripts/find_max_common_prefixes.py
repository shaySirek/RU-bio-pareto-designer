import argparse
import sys
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def get_lcp(s1: str, s2: str) -> str:
    m_len = min(len(s1), len(s2))
    for i in range(m_len):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[:m_len]


def cluster_sequences(root_dir: Path, alpha: float) -> dict[str, list[dict]]:
    data = []
    for file_path in root_dir.glob("*_sequence.txt"):
        try:
            content = file_path.read_text().strip()
            rank = file_path.name.split("_")[0]
            data.append({"name": file_path.name, "rank": rank, "seq": content})
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    if not data:
        return {}

    data.sort(key=lambda x: x["seq"])

    clusters = {}
    visited = [False] * len(data)

    for i in range(len(data)):
        if visited[i]:
            continue

        current_seq = data[i]["seq"]
        best_prefix = ""
        current_group = [data[i]]

        if i + 1 < len(data):
            neighbor_seq = data[i + 1]["seq"]
            lcp = get_lcp(current_seq, neighbor_seq)
            min_required = min(len(current_seq), len(neighbor_seq)) * alpha

            if len(lcp) >= min_required and len(lcp) > 0:
                best_prefix = lcp
                j = i + 1
                while j < len(data) and data[j]["seq"].startswith(best_prefix):
                    current_group.append(data[j])
                    visited[j] = True
                    j += 1

        key = best_prefix if best_prefix else "unclustered"
        if key not in clusters:
            clusters[key] = []
        clusters[key].extend(current_group)
        visited[i] = True

    return clusters


def save_rank_visualization(clusters: dict[str, list[dict]], output_path: Path):
    all_ranks = sorted(
        list(set(item["rank"] for group in clusters.values() for item in group))
    )
    if not all_ranks:
        return

    rank_to_idx = {rank: i for i, rank in enumerate(all_ranks)}
    n = len(all_ranks)
    matrix = np.zeros((n, n), dtype=int)

    for prefix, items in clusters.items():
        if prefix == "unclustered":
            continue
        cluster_ranks = [item["rank"] for item in items]
        counts = Counter(cluster_ranks)
        indices = [rank_to_idx[r] for r in counts.keys()]
        vals = np.array(list(counts.values()))
        matrix[np.ix_(indices, indices)] += np.outer(vals, vals)

    df = pd.DataFrame(matrix, index=all_ranks, columns=all_ranks)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=len(all_ranks) < 20, cmap="YlGnBu", fmt="d")
    plt.title("Rank Co-occurrence Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_mcp_scatter(clusters: dict[str, list[dict]], alpha: float, output_path: Path):
    plot_data = [
        (len(p), len(items)) for p, items in clusters.items() if p != "unclustered"
    ]
    if not plot_data:
        return

    lengths, sizes = zip(*plot_data)
    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, sizes, color="teal", alpha=0.6, edgecolors="black", s=80)
    plt.title(f"Cluster Size vs. Prefix Length (alpha={alpha})")
    plt.xlabel("Max Common Prefix Length (bp)")
    plt.ylabel("Number of Solutions")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster sequences and visualize rank relationships."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("-a", "--alpha", type=float, default=0.5)

    args = parser.parse_args()
    root, alpha = args.root, args.alpha

    if not root.is_dir():
        sys.exit(1)

    results = cluster_sequences(root, alpha)

    for prefix, items in results.items():
        if prefix != "unclustered":
            print(f"{len(items)} solutions have prefix of {len(prefix)} bp")
        else:
            print(f"{len(items)} solutions are unclustered")

    save_mcp_scatter(results, alpha, root / "mcp_scatter.png")
    save_rank_visualization(results, root / "rank_cooccurrence.png")

from typing import Optional, Union
from enum import Enum

from loguru import logger
import numpy as np
from sklearn.cluster import KMeans
import kmeans1d

PartitioningMap = dict[str, set[str]]
PartitioningResult = tuple[PartitioningMap, float]


class PartitioningMethod(Enum):
    KMEANS_1D = "kmeans1d"
    KMEANS = "kmeans"
    KMEANS_POLAR = "kmeans-polar_2"

    def partition(self, scores: np.ndarray, n_colors: int) -> np.ndarray:
        fn = {
            PartitioningMethod.KMEANS_1D: kmeans_1d,
            PartitioningMethod.KMEANS: kmeans,
            PartitioningMethod.KMEANS_POLAR: kmeans_polar,
        }[self]
        return fn(scores, n_colors)


def print_partitioning(
    scores: np.ndarray,
    n_colors: int,
    scores_bin: np.ndarray,
    title: str = "",
    file_path: Optional[str] = None,
):
    s_list = []

    def print_and_add_to_buf(line: str):
        logger.info(line)
        s_list.append(line + "\n")

    if len(title) > 0:
        print_and_add_to_buf(f"\n{title}")
    for i in range(n_colors):
        subset_scores = scores[np.where(scores_bin == i)]
        bin_size = subset_scores.shape[0]
        display = f"{subset_scores.shape[0]:10d} patterns"
        if bin_size > 0:
            display += f"\t[{np.min(subset_scores):.3f}, {np.max(subset_scores):.3f}]"
        print_and_add_to_buf(f"\tBin {i:2d}: {display}")

    if file_path is not None:
        with open(file_path, "a") as f:
            f.writelines(s_list)


def kmeans_1d(scores: np.ndarray, n_colors: int) -> np.ndarray:
    clusters, _ = kmeans1d.cluster(scores.tolist(), n_colors)
    scores_bin = np.array(clusters)
    return scores_bin


def kmeans(scores: np.ndarray, n_colors: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init="auto")
    scores_bin = kmeans.fit_predict(scores.reshape((-1, 1)))
    return scores_bin


def kmeans_polar(scores: np.ndarray, n_colors: int) -> np.ndarray:
    """Splits scores into positive & negative, applies KMeans separately to each."""
    pos_scores = scores[scores >= 0]
    neg_scores = scores[scores < 0]

    neg_n_colors = n_colors // 3
    pos_n_colors = n_colors - neg_n_colors
    pos_bins = kmeans(pos_scores, pos_n_colors) if len(pos_scores) > 0 else np.array([])
    neg_bins = kmeans(neg_scores, neg_n_colors) if len(neg_scores) > 0 else np.array([])

    # Combine bins for consistent coloring
    scores_bin = np.zeros_like(scores, dtype=int)
    scores_bin[scores < 0] = neg_bins
    scores_bin[scores >= 0] = pos_bins + neg_n_colors

    return scores_bin


def partition_to_colors(
    binding_score_map: dict[str, float],
    n_colors: int,
    method: PartitioningMethod,
    origin_n_colors: Optional[int] = None,
    return_baseline: bool = False,
    cmp_file_path: Optional[str] = None,
) -> Union[PartitioningResult, tuple[PartitioningResult, PartitioningResult]]:
    origin_n_colors = origin_n_colors or n_colors

    logger.info(
        f"Partitioning {len(binding_score_map)} patterns into {origin_n_colors} colors using {method.value}"
    )
    patterns = np.array(list(binding_score_map.keys()), dtype=str)
    scores = np.array(list(binding_score_map.values()), dtype=np.float32)
    scores_bin = method.partition(scores, origin_n_colors)
    print_partitioning(scores, origin_n_colors, scores_bin, file_path=cmp_file_path)

    # sub-partitioning case
    if origin_n_colors < n_colors:
        logger.info(
            f"Transform partitioning from {origin_n_colors} colors to {n_colors} colors..."
        )
        scores_bin = split_partitions(scores, scores_bin, n_colors, method)
        print_partitioning(
            scores,
            n_colors,
            scores_bin,
            title="After spliting partitions:",
            file_path=cmp_file_path,
        )

    # merge-subsets case
    elif origin_n_colors > n_colors:
        logger.info(
            f"Transform partitioning from {origin_n_colors} colors to {n_colors} colors..."
        )
        scores_bin = join_partitions(scores, scores_bin, n_colors)
        print_partitioning(
            scores,
            n_colors,
            scores_bin,
            title="After joining partitions:",
            file_path=cmp_file_path,
        )

    result = get_partitioning_result(binding_score_map, patterns, scores_bin, n_colors)

    if origin_n_colors != n_colors:
        baseline_scores_bin = method.partition(scores, n_colors)
        print_partitioning(
            scores,
            n_colors,
            baseline_scores_bin,
            title="Baseline:",
            file_path=cmp_file_path,
        )
        if return_baseline:
            baseline_result = get_partitioning_result(
                binding_score_map, patterns, baseline_scores_bin, n_colors
            )
            result = (result, baseline_result)

        cmp_file_path = cmp_file_path or f"cmp_{origin_n_colors}---{n_colors}"
        compare_partitions(
            scores, scores_bin, baseline_scores_bin, patterns, cmp_file_path
        )

    return result


def split_partitions(
    scores: np.ndarray,
    scores_bin: np.ndarray,
    target_n_colors: int,
    method: PartitioningMethod,
) -> np.ndarray:
    scores_bin = scores_bin.copy()
    uniq_bins = np.unique(scores_bin)

    while len(uniq_bins) < target_n_colors:
        chosen_bin = max(uniq_bins, key=lambda b: np.var(scores[scores_bin == b]))
        in_bin_mask = scores_bin == chosen_bin
        scores_in_bin = scores[in_bin_mask]

        logger.info(
            f"\n\tSplitting bin {chosen_bin} (size={in_bin_mask.sum()}) into 2 bins"
        )
        new_bins = method.partition(scores_in_bin, 2)
        scores_bin[scores_bin > chosen_bin] += 1
        scores_bin[in_bin_mask] = np.where(new_bins == 0, chosen_bin, chosen_bin + 1)
        uniq_bins, scores_bin = np.unique(scores_bin, return_inverse=True)

    return scores_bin


def join_partitions(
    scores: np.ndarray,
    scores_bin: np.ndarray,
    target_n_colors: int,
) -> np.ndarray:
    scores_bin = scores_bin.copy()
    uniq_bins = np.unique(scores_bin)

    while len(uniq_bins) > target_n_colors:
        b1, b2 = min(
            zip(uniq_bins[:-1], uniq_bins[1:]),
            key=lambda pair: (
                np.var(scores[(scores_bin == pair[0]) | (scores_bin == pair[1])])
                - np.var(scores[scores_bin == pair[0]])
                - np.var(scores[scores_bin == pair[1]])
            ),
        )
        logger.info(f"\n\tMerging bins {b1:2d} and {b2:2d}")
        scores_bin[scores_bin == b2] = b1
        scores_bin[scores_bin > b2] -= 1
        uniq_bins = np.unique(scores_bin)

    return scores_bin


def get_partitioning_result(
    binding_score_map: dict[str, float],
    patterns: np.ndarray,
    scores_bin: np.ndarray,
    n_colors: int,
) -> PartitioningResult:
    colored_patterns = patterns_per_bin(patterns, scores_bin, n_colors)
    sse = sum(
        len(subset_patterns) * np.var([binding_score_map[v] for v in subset_patterns])
        for subset_patterns in colored_patterns.values()
    )
    logger.info(
        f"\nSSE          = {sse:.3f}"
        f"\nSSE / |V|    = {sse / len(binding_score_map):.6f}"
    )

    return colored_patterns, sse


def patterns_per_bin(
    patterns: np.ndarray,
    scores_bin: np.ndarray,
    n_colors: int,
) -> PartitioningMap:
    bin_patterns: PartitioningMap = {}
    total = 0

    for i in range(n_colors):
        bin_patterns[str(i)] = set(patterns[np.where(scores_bin == i)].tolist())
        bin_size = len(bin_patterns[str(i)])
        total += bin_size

    assert total == patterns.shape[0]

    return bin_patterns


def compare_partitions(
    scores: np.ndarray,
    bin1: np.ndarray,
    bin2: np.ndarray,
    patterns: np.ndarray,
    output_file_path: str,
):
    assert bin1.shape == bin2.shape, "Partitionings must have the same shape"
    assert len(np.unique(bin1)) == len(
        np.unique(bin2)
    ), "Must have the same number of bins"

    with open(output_file_path, "a") as f:
        f.write(100 * ">" + "\n")
        n_bins = len(np.unique(bin1))
        f.write(f"Comparing partitioning with {n_bins} bins...\n")

        # Overall disagreement
        mismatch = bin1 != bin2
        mismatch_count = np.sum(mismatch)
        f.write(f"Total differing assignments: {mismatch_count} / {len(scores)}\n")

        overlap_matrix = np.zeros((n_bins, n_bins), dtype=int)
        for i in range(n_bins):
            for j in range(n_bins):
                overlap_matrix[i, j] = np.sum((bin1 == i) & (bin2 == j))
                if i != j and overlap_matrix[i, j] > 0:
                    # Find indices where bin1 has 'i' but bin2 is different
                    diff_idx = np.where((bin1 == i) & (bin2 != i))[0]
                    # Filter patterns corresponding to these differing indices
                    diff = patterns[diff_idx].tolist()
                    if len(diff) > 0:
                        p_display = "\n\t".join(diff)
                        f.write(
                            f"\n\tBin {i} has {len(diff)} more patterns compared to the baseline"
                            f"\n\t{p_display}\n\t.............\n"
                        )

        f.write("\nPer-bin overlap matrix (alternative vs baseline):\n")
        f.write(str(overlap_matrix) + "\n")
        f.write(100 * ">" + "\n")

    logger.info(f"Comparison report successfully written to '{output_file_path}'")

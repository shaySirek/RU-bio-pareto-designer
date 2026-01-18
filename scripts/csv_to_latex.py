"""
Usage:
    python3.10 .\scripts\csv_to_latex.py .\experiments\base_kmeans1d.csv base_kmeans1d_2_latex \
        --partition-column motif_length \
        --multirow-column matrix_id \
        --include-columns matrix_id,number_of_colors,n_states_irreducible_fsm,binding_mse
"""

import argparse
import os
from collections import deque
import re

import pandas as pd


def escape_latex(s):
    if isinstance(s, str):
        return (
            s.replace("&", r"\&")
            .replace("%", r"\%")
            .replace("$", r"\$")
            .replace("#", r"\#")
            .replace("_", r"\_")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("~", r"\textasciitilde{}")
            .replace("^", r"\textasciicircum{}")
            .replace("\\", r"\textbackslash{}")
        )
    return s


def apply_multirow(df, multirow_column, paired_columns=None):
    df = df.copy()
    if paired_columns is None:
        paired_columns = []

    all_multirow_columns = [multirow_column] + paired_columns
    df[all_multirow_columns] = (
        df[all_multirow_columns].applymap(str).applymap(escape_latex)
    )

    values = df[multirow_column].values
    rows = []
    skip = deque()

    for i in range(len(df)):
        if skip and skip[0] == i:
            skip.popleft()
            continue

        span = 1
        for j in range(i + 1, len(df)):
            if values[j] == values[i]:
                span += 1
            else:
                break
        if span > 1:
            for k in range(1, span):
                skip.append(i + k)

            multirow_vals = {
                col: f"\\multirow{{{span}}}{{*}}{{{df.iloc[i][col]}}}"
                for col in all_multirow_columns
            }
        else:
            multirow_vals = {col: df.iloc[i][col] for col in all_multirow_columns}

        row = df.iloc[i].copy()
        for col in all_multirow_columns:
            row[col] = multirow_vals[col]
        rows.append(row)

        for k in range(1, span):
            next_row = df.iloc[i + k].copy()
            for col in all_multirow_columns:
                next_row[col] = ""
            rows.append(next_row)

    return pd.DataFrame(rows, columns=df.columns)


def format_col_name(col: str) -> str:
    return (
        col.replace("n_states", "number_of_states_in")
        .replace("_", " ")
        .title()
        .replace("Db", "DB")
        .replace("Fsm", "FSM")
        .replace("Mse", "Score MSE")
        .replace("Id", "ID")
        .replace("Number Of", r"\#")
        .replace("In", "in")
        .replace("Perc", r" \%")
        .replace("Sec", " (sec.)")
    )


def wrap_with_tabularx(latex: str, num_columns: int, caption: str, label: str) -> str:
    # Safely replace the \begin{tabular}{...} with tabularx and Y columns
    latex = re.sub(
        r"\\begin{tabular}{[^}]*}",
        rf"\\begin{{tabularx}}{{\\linewidth}}{{{'Y' * num_columns}}}",
        latex,
    )
    latex = re.sub(
        r"(\\\\)\s*\n(\s*\\multirow)",  # Match \\ followed by newline and spaces, then \multirow
        r"\1\\addlinespace\n\2",  # Insert \addlinespace between
        latex,
    )
    latex = latex.replace(r"\end{tabular}", r"\end{tabularx}")

    # Wrap the whole thing in a table environment
    wrapped = (
        "\\newpage\n"
        "\\thispagestyle{empty}\n"
        "\\begin{table}[H]\n"
        "\\centering\n" + latex + "\n\\caption{" + caption + "}\n"
        "\\label{tab:" + label + "}\n"
        "\\end{table}"
    )

    return wrapped


def csv_to_latex(
    csv_file,
    output_file=None,
    index=False,
    partition_column=None,
    include_cols=None,
    multirow_column=None,
    multirow_paired_columns=None,
):

    df = pd.read_csv(csv_file)
    df = df[~df["matrix_id"].str.contains(".rc")]
    df = df[
        ~df["matrix_id"].str.contains(
            "|".join(
                [
                    "MA0210.1",
                    "MA0218.1",
                    "MA0026.1",
                ]
            )
        )
    ]
    df.columns = [format_col_name(col) for col in df.columns]

    if include_cols:
        missing = [col for col in include_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Cannot use non existing columns: {missing}")
        if partition_column in df.columns:
            include_cols.append(partition_column)
        df = df[include_cols]

    if partition_column:
        if partition_column not in df.columns:
            raise ValueError(f"Partition column '{partition_column}' not found.")
        grouped = df.groupby(partition_column)
        base_name = os.path.splitext(output_file or csv_file)[0]
        base_dir = f"{base_name}_tables/{partition_column.replace(' ', '_')}"
        os.makedirs(base_dir, exist_ok=True)

        for key, group_df in grouped:
            group_df = group_df.drop(columns=[partition_column])
            multi_count = group_df[multirow_column].unique().shape[0]
            if multirow_column:
                group_df = apply_multirow(
                    group_df, multirow_column, multirow_paired_columns
                )
            latex = group_df.to_latex(index=index, escape=False)
            latex = wrap_with_tabularx(
                latex,
                len(group_df.columns) + (1 if index else 0),
                caption=f"Statistics on irreducible FSMs computed for {multi_count} binding motifs of length $m={key}$."
                " The DB FSM of each motif was colored with $c$ colors, where $c\in\{3,6,9,15\}$, by applying the one-dimensional k-means algorithm (see Section \\ref{sec:coloring_and_scoring})."
                " Then, the irreducible FSM of this colored FSM was computed and we report the number of states in this FSM and its MSE."
                f" The number of states in the original DB-FSM is $4^m$ = {4**int(key)} for all motifs.",
                label=f"irreducible_fsms_results_len_{key}",
            )
            safe_key = str(key).replace(" ", "_").replace("/", "_")
            out_file = f"{base_dir}/{safe_key}.tex"
            with open(out_file, "w") as f:
                f.write(latex)
            print(f"Wrote LaTeX table for group '{key}' to {out_file}")
    else:
        if multirow_column:
            df = apply_multirow(df, multirow_column, multirow_paired_columns)
        latex = df.to_latex(index=index, escape=False)
        latex = wrap_with_tabularx(
            latex,
            len(df.columns) + (1 if index else 0),
            caption="Irreducible FSMs colored with different amounts of colors",
            label="irreducible_fsms_results",
        )
        if output_file:
            with open(output_file, "w") as f:
                f.write(latex)
            print(f"Wrote LaTeX table to {output_file}")
        else:
            print(latex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to LaTeX table(s).")
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument(
        "output_file", nargs="?", help="Output LaTeX file or prefix if partitioning."
    )
    parser.add_argument("--index", action="store_true", help="Include DataFrame index.")
    parser.add_argument(
        "--partition-column",
        help="Group by this column and export separate LaTeX files.",
    )
    parser.add_argument("--include-columns", help="Comma-separated columns to include.")
    parser.add_argument(
        "--multirow-column", help="Column to collapse using LaTeX \\multirow."
    )
    parser.add_argument(
        "--multirow-paired-columns",
        help="Comma-separated list of columns that depend only on --multirow-column.",
    )

    args = parser.parse_args()

    include_cols = (
        [format_col_name(c.strip()) for c in args.include_columns.split(",")]
        if args.include_columns
        else None
    )
    part_col = format_col_name(args.partition_column) if args.partition_column else None
    multirow_col = (
        format_col_name(args.multirow_column) if args.multirow_column else None
    )
    multirow_pairs = (
        [format_col_name(c.strip()) for c in args.multirow_paired_columns.split(",")]
        if args.multirow_paired_columns
        else None
    )

    csv_to_latex(
        csv_file=args.csv_file,
        output_file=args.output_file,
        index=args.index,
        partition_column=part_col,
        include_cols=include_cols,
        multirow_column=multirow_col,
        multirow_paired_columns=multirow_pairs,
    )

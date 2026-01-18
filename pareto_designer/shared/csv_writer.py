from typing import Iterator, Any

import csv
from pathlib import Path


def write_results_stream(
    results_generator: Iterator[dict[str, Any]], filepath: Path
) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="") as csvfile:
        writer: csv.DictWriter | None = None
        for i, row in enumerate(results_generator):
            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            csvfile.flush()

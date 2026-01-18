import re
from pathlib import Path
from loguru import logger

MOTIF_ID = "MA0267.1"
BASE_DIR = Path("bio_data")
PATTERN_FILE = BASE_DIR / "motifs" / MOTIF_ID / "significant_patterns.txt"
SEQUENCE_DIR = BASE_DIR / "zea_mays_genes"
OUTPUT_DIR = BASE_DIR / "hits" / MOTIF_ID


def load_patterns(path: Path):
    if not path.exists():
        logger.error(f"Pattern file missing: {path}")
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def find_all_overlaps(sequence, pattern):
    matches = []
    for m in re.finditer(f"(?=({pattern}))", sequence):
        start = m.start()
        content = m.group(1)
        matches.append(
            {"pattern": content, "start": start + 1, "stop": start + len(content)}
        )
    return matches


def process_sequences(motif: str, p_file: Path, s_dir: Path, out_dir: Path):
    patterns = load_patterns(p_file)
    if not patterns or not s_dir.exists():
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for seq_path in sorted(s_dir.glob("*.txt")):
        strand_match = re.search(r"_(-?\d)_", seq_path.name)
        strand_info = (
            ("+" if strand_match.group(1) == "1" else "-")
            if strand_match
            else "Unknown"
        )

        sequence = seq_path.read_text().strip()
        hits = []
        for p in patterns:
            hits.extend(find_all_overlaps(sequence, p))

        hits.sort(key=lambda x: x["start"])

        if hits:
            out_file = out_dir / f"{seq_path.stem}_hits.txt"
            with out_file.open("w") as f:
                f.write(f"FILE: {seq_path.name}\n")
                f.write(f"MOTIF: {motif} | STRAND: {strand_info}\n")
                f.write(f"{'Pattern':<15} {'Start':<8} {'Stop':<8}\n")
                f.write("-" * 35 + "\n")

                for h in hits:
                    f.write(f"{h['pattern']:<15} {h['start']:<8} {h['stop']:<8}\n")

                f.write("\nOVERLAPS:\n")
                for i in range(len(hits) - 1):
                    curr, nxt = hits[i], hits[i + 1]
                    if curr["stop"] >= nxt["start"]:
                        f.write(
                            f"  {curr['pattern']}({curr['start']}) with {nxt['pattern']}({nxt['start']})\n"
                        )

            logger.info(f"Saved hits to: {out_file.name}")


if __name__ == "__main__":
    process_sequences(MOTIF_ID, PATTERN_FILE, SEQUENCE_DIR, OUTPUT_DIR)

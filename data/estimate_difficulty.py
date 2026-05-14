"""Assign difficulty buckets from teacher long-CoT length.

Run from the project root:
python data/estimate_difficulty.py
"""

from pathlib import Path

from data_utils import read_jsonl, write_jsonl


BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "cold_start" / "teacher_long_gsm8k_train.filtered.jsonl"
OUTPUT_FILE = BASE_DIR / "cold_start" / "teacher_long_gsm8k_train.difficulty.jsonl"


def count_tokens(text):
    return len(str(text).split())


def difficulty_level(rank, total):
    if total <= 1:
        return "medium"

    fraction = rank / total
    if fraction < 1 / 3:
        return "easy"
    if fraction < 2 / 3:
        return "medium"
    return "hard"


def main():
    records = read_jsonl(INPUT_FILE)

    for record in records:
        record["difficulty_score"] = count_tokens(record["teacher_long_rationale"])

    sorted_records = sorted(records, key=lambda x: x["difficulty_score"])
    total = len(sorted_records)

    for rank, record in enumerate(sorted_records):
        record["difficulty_level"] = difficulty_level(rank, total)

    # Restore the original data order so downstream files are easy to compare.
    id_to_record = {}
    for record in sorted_records:
        id_to_record[record["id"]] = record
    output_records = [id_to_record[record["id"]] for record in records]

    write_jsonl(OUTPUT_FILE, output_records)

    counts = {"easy": 0, "medium": 0, "hard": 0}
    for record in output_records:
        counts[record["difficulty_level"]] += 1

    print(f"Read {len(records)} records from {INPUT_FILE}")
    print(f"Wrote difficulty-labeled records to {OUTPUT_FILE}")
    print(f"easy: {counts['easy']}")
    print(f"medium: {counts['medium']}")
    print(f"hard: {counts['hard']}")


if __name__ == "__main__":
    main()

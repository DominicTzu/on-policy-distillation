"""Load GSM8K and save raw/processed JSONL files.

Run from the project root:
python data/prepare_data.py

Change the constants below if you want another split or a smaller debug run.
"""

from pathlib import Path

from data_utils import load_gsm8k_records, to_raw_gsm8k_records, write_jsonl


DATASET = "gsm8k"
SUBSET = "main"
SPLIT = "train"

# Set to an integer like 100 for a quick debug run.
LIMIT = None

BASE_DIR = Path(__file__).parent
RAW_OUTPUT = BASE_DIR / "raw" / f"{DATASET}_{SPLIT}.jsonl"
PROCESSED_OUTPUT = BASE_DIR / "processed" / f"{DATASET}_{SPLIT}.jsonl"


def main():
    if DATASET != "gsm8k":
        raise ValueError("Only GSM8K is supported right now.")

    records = load_gsm8k_records(SPLIT, SUBSET, LIMIT)

    raw_count = write_jsonl(RAW_OUTPUT, to_raw_gsm8k_records(records))
    processed_count = write_jsonl(PROCESSED_OUTPUT, records)

    print(f"Wrote {raw_count} raw records to {RAW_OUTPUT}")
    print(f"Wrote {processed_count} processed records to {PROCESSED_OUTPUT}")


if __name__ == "__main__":
    main()

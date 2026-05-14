"""Filter teacher long-CoT generations before difficulty estimation.

Run from the project root:
python data/filter_long_cot.py
"""

from collections import Counter
from pathlib import Path

from answer_utils import (
    answers_match,
    extract_answer,
    normalize_answer,
    split_long_rationale_and_answer,
)
from data_utils import read_jsonl, write_jsonl


BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "cold_start" / "teacher_long_gsm8k_train.jsonl"
OUTPUT_FILE = BASE_DIR / "cold_start" / "teacher_long_gsm8k_train.filtered.jsonl"
REJECTS_FILE = BASE_DIR / "cold_start" / "teacher_long_gsm8k_train.rejects.jsonl"

MIN_LONG_RATIONALE_CHARS = 2
REQUIRE_CORRECT = True


def filter_long_record(record):
    response = str(record.get("teacher_long_response", "")).strip()
    rationale, answer = split_long_rationale_and_answer(response)

    if rationale is None or answer is None:
        return None, "missing_required_sections"
    if len(rationale.strip()) < MIN_LONG_RATIONALE_CHARS:
        return None, "empty_long_rationale"
    if not answer.strip():
        return None, "empty_answer"

    parsed_answer = extract_answer(response)
    normalized_answer = normalize_answer(parsed_answer)
    if normalized_answer is None:
        return None, "answer_parse_failed"

    teacher_correct = answers_match(parsed_answer, record.get("gold_answer"))
    if REQUIRE_CORRECT and not teacher_correct:
        return None, "answer_mismatch"

    filtered = dict(record)
    filtered["teacher_long_rationale"] = rationale.strip()
    filtered["teacher_long_answer"] = answer.strip()
    filtered["teacher_long_answer_parsed"] = parsed_answer
    filtered["teacher_long_answer_normalized"] = normalized_answer
    filtered["teacher_long_correct"] = teacher_correct
    return filtered, None


def main():
    records = read_jsonl(INPUT_FILE)
    kept = []
    rejects = []
    stats = Counter()

    for record in records:
        filtered, reason = filter_long_record(record)
        if filtered is None:
            rejected = dict(record)
            rejected["filter_reason"] = reason
            rejects.append(rejected)
            stats[reason] += 1
        else:
            kept.append(filtered)
            stats["kept"] += 1

    write_jsonl(OUTPUT_FILE, kept)
    write_jsonl(REJECTS_FILE, rejects)

    print(f"Read {len(records)} records from {INPUT_FILE}")
    print(f"Wrote {len(kept)} filtered long-CoT records to {OUTPUT_FILE}")
    print(f"Wrote {len(rejects)} rejected records to {REJECTS_FILE}")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

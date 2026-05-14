"""Filter compressed rationales and build the cold-start SFT dataset.

Run from the project root:
python data/filtering.py
"""

from collections import Counter
from pathlib import Path

from answer_utils import (
    answers_match,
    extract_answer,
    normalize_answer,
    split_rationale_and_answer,
)
from data_utils import read_jsonl, write_jsonl


BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "cold_start" / "teacher_compressed_gsm8k_train.jsonl"
OUTPUT_FILE = BASE_DIR / "cold_start" / "cold_start_gsm8k_train.jsonl"
REJECTS_FILE = BASE_DIR / "cold_start" / "teacher_compressed_gsm8k_train.rejects.jsonl"
STATS_FILE = BASE_DIR / "cold_start" / "compression_stats.jsonl"

MIN_RATIONALE_CHARS = 2
MAX_RATIONALE_WORDS = 160

# README marks this as optional. Keep it off by default.
CHECK_MIN_MEDIUM_HARD_LENGTH = False
MIN_MEDIUM_HARD_WORDS = 6


def count_tokens(text):
    return len(str(text).split())


def add_compression_stats(record):
    long_tokens = count_tokens(record.get("teacher_long_rationale", ""))
    compressed_tokens = count_tokens(record.get("teacher_compressed_rationale", ""))

    record["long_rationale_tokens"] = long_tokens
    record["compressed_rationale_tokens"] = compressed_tokens

    if long_tokens > 0:
        ratio = compressed_tokens / long_tokens
        record["compression_ratio"] = ratio
        record["token_reduction"] = 1 - ratio
    else:
        record["compression_ratio"] = None
        record["token_reduction"] = None


def filter_compressed_record(record):
    response = str(record.get("teacher_compressed_response", "")).strip()
    rationale, answer = split_rationale_and_answer(response)

    if rationale is None or answer is None:
        return None, "missing_required_sections"
    if len(rationale.strip()) < MIN_RATIONALE_CHARS:
        return None, "empty_rationale"
    if not answer.strip():
        return None, "empty_answer"

    parsed_answer = extract_answer(response)
    normalized_answer = normalize_answer(parsed_answer)
    if normalized_answer is None:
        return None, "answer_parse_failed"

    teacher_correct = answers_match(parsed_answer, record.get("gold_answer"))
    if not teacher_correct:
        return None, "answer_mismatch"

    filtered = dict(record)
    filtered["teacher_compressed_rationale"] = rationale.strip()
    filtered["teacher_compressed_answer"] = answer.strip()
    filtered["teacher_compressed_answer_parsed"] = parsed_answer
    filtered["teacher_compressed_answer_normalized"] = normalized_answer
    filtered["teacher_compressed_correct"] = teacher_correct
    add_compression_stats(filtered)

    if filtered["long_rationale_tokens"] <= 0:
        return None, "missing_long_rationale"
    if filtered["compressed_rationale_tokens"] >= filtered["long_rationale_tokens"]:
        return None, "not_compressed"
    if filtered["compressed_rationale_tokens"] > MAX_RATIONALE_WORDS:
        return None, "compressed_rationale_too_long"

    level = filtered.get("difficulty_level")
    if CHECK_MIN_MEDIUM_HARD_LENGTH and level in ["medium", "hard"]:
        if filtered["compressed_rationale_tokens"] < MIN_MEDIUM_HARD_WORDS:
            return None, "compressed_rationale_too_short"

    filtered["target_response"] = (
        "### Rationale\n"
        + filtered["teacher_compressed_rationale"]
        + "\n\n### Answer\n"
        + filtered["teacher_compressed_answer"]
    )
    return filtered, None


def summarize(records):
    groups = {"overall": records, "easy": [], "medium": [], "hard": []}
    for record in records:
        level = record.get("difficulty_level")
        if level in groups:
            groups[level].append(record)

    summary = []
    for name, group in groups.items():
        if not group:
            summary.append(
                {
                    "bucket": name,
                    "num_examples": 0,
                    "avg_long_rationale_tokens": 0.0,
                    "avg_compressed_rationale_tokens": 0.0,
                    "avg_compression_ratio": 0.0,
                    "avg_token_reduction": 0.0,
                }
            )
            continue

        summary.append(
            {
                "bucket": name,
                "num_examples": len(group),
                "avg_long_rationale_tokens": sum(
                    r["long_rationale_tokens"] for r in group
                )
                / len(group),
                "avg_compressed_rationale_tokens": sum(
                    r["compressed_rationale_tokens"] for r in group
                )
                / len(group),
                "avg_compression_ratio": sum(r["compression_ratio"] for r in group)
                / len(group),
                "avg_token_reduction": sum(r["token_reduction"] for r in group)
                / len(group),
            }
        )
    return summary


def main():
    records = read_jsonl(INPUT_FILE)
    kept = []
    rejects = []
    stats = Counter()

    for record in records:
        filtered, reason = filter_compressed_record(record)
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
    write_jsonl(STATS_FILE, summarize(kept))

    print(f"Read {len(records)} records from {INPUT_FILE}")
    print(f"Wrote {len(kept)} cold-start records to {OUTPUT_FILE}")
    print(f"Wrote {len(rejects)} rejected records to {REJECTS_FILE}")
    print(f"Wrote compression stats to {STATS_FILE}")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

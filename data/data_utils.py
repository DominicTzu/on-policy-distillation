"""Shared data loading and JSONL helpers."""

import json
from pathlib import Path

from answer_utils import extract_gsm8k_gold_answer, extract_gsm8k_rationale


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    ensure_parent_dir(path)
    count = 0
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path, record):
    ensure_parent_dir(path)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_gsm8k_records(split, subset, limit):
    from datasets import load_dataset

    dataset = load_dataset("gsm8k", subset, split=split)
    records = []

    for idx, example in enumerate(dataset):
        if limit is not None and idx >= limit:
            break

        solution = str(example["answer"])
        records.append(
            {
                "id": f"gsm8k-{split}-{idx}",
                "dataset": "gsm8k",
                "split": split,
                "question": str(example["question"]),
                "gold_solution": solution,
                "gold_rationale": extract_gsm8k_rationale(solution),
                "gold_answer": extract_gsm8k_gold_answer(solution),
            }
        )

    return records


def to_raw_gsm8k_records(records):
    raw_records = []
    for record in records:
        raw_records.append(
            {
                "id": record["id"],
                "dataset": record["dataset"],
                "split": record["split"],
                "question": record["question"],
                "answer": record["gold_solution"],
            }
        )
    return raw_records

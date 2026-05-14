"""Shared helpers for Stage B on-policy distillation."""

import json
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from answer_utils import (  # noqa: E402
    answers_match,
    extract_answer,
    normalize_answer,
    split_rationale_and_answer,
)


PROMPT_TEMPLATE = "### Question\n{question}\n\n"
RESPONSE_TEMPLATE = "### Rationale\n{rationale}\n\n### Answer\n{answer}"


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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def progress(items, desc=None):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc)
    except ImportError:
        return items


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_torch_dtype(device, dtype_name):
    if dtype_name == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device == "cuda":
            return torch.float16
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def get_model_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def build_prompt(record):
    return PROMPT_TEMPLATE.format(question=str(record["question"]).strip())


def count_tokens(text):
    return len(str(text or "").split())


def load_latest_checkpoint(latest_checkpoint_file, fallback_checkpoint):
    latest_checkpoint_file = Path(latest_checkpoint_file)
    if latest_checkpoint_file.exists():
        checkpoint = latest_checkpoint_file.read_text(encoding="utf-8").strip()
        if checkpoint:
            return Path(checkpoint)
    return Path(fallback_checkpoint)


def load_completed_ids(path):
    path = Path(path)
    if not path.exists():
        return set()
    completed_ids = set()
    for record in iter_jsonl(path):
        if "id" in record:
            completed_ids.add(str(record["id"]))
    return completed_ids


def parse_student_response(response, gold_answer):
    rationale, answer = split_rationale_and_answer(response)
    parsed_answer = extract_answer(response)
    normalized_answer = normalize_answer(parsed_answer)

    return {
        "student_rationale": rationale,
        "student_answer": answer,
        "student_answer_parsed": parsed_answer,
        "student_answer_normalized": normalized_answer,
        "student_correct": answers_match(parsed_answer, gold_answer),
        "student_parse_failed": normalized_answer is None,
        "student_missing_required_sections": rationale is None or answer is None,
        "student_response_tokens": count_tokens(response),
        "student_rationale_tokens": count_tokens(rationale),
    }


def summarize_generation_records(records):
    def summarize_group(name, group):
        num_examples = len(group)
        if num_examples == 0:
            return {
                "bucket": name,
                "num_examples": 0,
                "accuracy": 0.0,
                "parse_fail_rate": 0.0,
                "avg_response_tokens": 0.0,
                "avg_rationale_tokens": 0.0,
            }

        correct = sum(1 for record in group if record.get("student_correct"))
        parse_failed = sum(1 for record in group if record.get("student_parse_failed"))
        return {
            "bucket": name,
            "num_examples": num_examples,
            "accuracy": correct / num_examples,
            "parse_fail_rate": parse_failed / num_examples,
            "avg_response_tokens": sum(
                record.get("student_response_tokens", 0) for record in group
            )
            / num_examples,
            "avg_rationale_tokens": sum(
                record.get("student_rationale_tokens", 0) for record in group
            )
            / num_examples,
        }

    overall = summarize_group("overall", records)
    by_difficulty = []
    for level in ["easy", "medium", "hard"]:
        group = [record for record in records if record.get("difficulty_level") == level]
        if group:
            by_difficulty.append(summarize_group(level, group))

    summary = {
        "num_examples": overall["num_examples"],
        "accuracy": overall["accuracy"],
        "parse_fail_rate": overall["parse_fail_rate"],
        "avg_response_tokens": overall["avg_response_tokens"],
        "avg_rationale_tokens": overall["avg_rationale_tokens"],
        "overall": overall,
    }
    if by_difficulty:
        summary["by_difficulty"] = by_difficulty
    return summary


def build_eval_metric_schema():
    return {
        "final_answer_accuracy": "accuracy",
        "parse_fail_rate": "parse_fail_rate",
        "avg_response_tokens": "avg_response_tokens",
        "avg_rationale_tokens": "avg_rationale_tokens",
        "difficulty_bucket_metrics": ["easy", "medium", "hard"],
        "error_recovery_inputs": [
            "cold_start_checkpoint",
            "on_policy_checkpoint",
            "same_evaluation_file",
        ],
    }

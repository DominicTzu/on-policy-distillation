"""Evaluation helpers for final-answer accuracy and reporting."""

import csv
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


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


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


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def get_model_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def load_latest_checkpoint(latest_checkpoint_file, fallback_checkpoint):
    latest_checkpoint_file = Path(latest_checkpoint_file)
    if latest_checkpoint_file.exists():
        checkpoint = latest_checkpoint_file.read_text(encoding="utf-8").strip()
        if checkpoint:
            return Path(checkpoint)
    return Path(fallback_checkpoint)


def build_prompt(record):
    return PROMPT_TEMPLATE.format(question=str(record["question"]).strip())


def count_tokens(text):
    return len(str(text or "").split())


def parse_response(response, gold_answer):
    rationale, answer = split_rationale_and_answer(response)
    parsed_answer = extract_answer(response)
    normalized_answer = normalize_answer(parsed_answer)
    return {
        "response": response,
        "rationale": rationale,
        "answer": answer,
        "answer_parsed": parsed_answer,
        "answer_normalized": normalized_answer,
        "correct": answers_match(parsed_answer, gold_answer),
        "parse_failed": normalized_answer is None,
        "missing_required_sections": rationale is None or answer is None,
        "response_tokens": count_tokens(response),
        "rationale_tokens": count_tokens(rationale),
    }


def summarize_results(results, checkpoint_name, checkpoint_path):
    num_examples = len(results)
    correct = sum(1 for record in results if record.get("correct"))
    parse_failed = sum(1 for record in results if record.get("parse_failed"))

    summary = {
        "checkpoint_name": checkpoint_name,
        "checkpoint": str(checkpoint_path),
        "num_examples": num_examples,
        "accuracy": correct / num_examples if num_examples else 0.0,
        "parse_fail_rate": parse_failed / num_examples if num_examples else 0.0,
        "avg_response_tokens": sum(
            record.get("response_tokens", 0) for record in results
        )
        / num_examples
        if num_examples
        else 0.0,
        "avg_rationale_tokens": sum(
            record.get("rationale_tokens", 0) for record in results
        )
        / num_examples
        if num_examples
        else 0.0,
    }

    difficulty = {}
    for level in ["easy", "medium", "hard"]:
        group = [record for record in results if record.get("difficulty_level") == level]
        if not group:
            continue
        group_correct = sum(1 for record in group if record.get("correct"))
        group_parse_failed = sum(1 for record in group if record.get("parse_failed"))
        difficulty[level] = {
            "num_examples": len(group),
            "accuracy": group_correct / len(group),
            "parse_fail_rate": group_parse_failed / len(group),
            "avg_response_tokens": sum(
                record.get("response_tokens", 0) for record in group
            )
            / len(group),
            "avg_rationale_tokens": sum(
                record.get("rationale_tokens", 0) for record in group
            )
            / len(group),
        }
    if difficulty:
        summary["difficulty"] = difficulty

    return summary


def compute_error_recovery(before_results, after_results):
    before_by_id = {str(record["id"]): record for record in before_results}
    after_by_id = {str(record["id"]): record for record in after_results}
    shared_ids = sorted(set(before_by_id) & set(after_by_id))

    wrong_before = 0
    wrong_before_correct_after = 0
    regressed_after = 0

    for record_id in shared_ids:
        before_correct = bool(before_by_id[record_id].get("correct"))
        after_correct = bool(after_by_id[record_id].get("correct"))
        if not before_correct:
            wrong_before += 1
            if after_correct:
                wrong_before_correct_after += 1
        elif not after_correct:
            regressed_after += 1

    return {
        "num_shared_examples": len(shared_ids),
        "wrong_before": wrong_before,
        "wrong_before_correct_after": wrong_before_correct_after,
        "error_recovery_rate": wrong_before_correct_after / wrong_before
        if wrong_before
        else 0.0,
        "correct_before_wrong_after": regressed_after,
        "regression_rate_among_correct_before": regressed_after
        / (len(shared_ids) - wrong_before)
        if len(shared_ids) - wrong_before > 0
        else 0.0,
    }


def format_pct(value):
    return f"{100 * float(value):.2f}%"


def format_float(value):
    return f"{float(value):.2f}"


def markdown_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def build_report(summary, error_recovery):
    overall_rows = []
    for item in summary["checkpoints"]:
        overall_rows.append(
            [
                item["checkpoint_name"],
                item["num_examples"],
                format_pct(item["accuracy"]),
                format_pct(item["parse_fail_rate"]),
                format_float(item["avg_response_tokens"]),
                format_float(item["avg_rationale_tokens"]),
            ]
        )

    report = [
        "# Evaluation Report",
        "",
        "## Overall",
        "",
        markdown_table(
            [
                "checkpoint",
                "num_examples",
                "accuracy",
                "parse_fail_rate",
                "avg_response_tokens",
                "avg_rationale_tokens",
            ],
            overall_rows,
        ),
    ]

    if any("difficulty" in item for item in summary["checkpoints"]):
        difficulty_rows = []
        for item in summary["checkpoints"]:
            for level in ["easy", "medium", "hard"]:
                metrics = item.get("difficulty", {}).get(level)
                if metrics is None:
                    continue
                difficulty_rows.append(
                    [
                        item["checkpoint_name"],
                        level,
                        metrics["num_examples"],
                        format_pct(metrics["accuracy"]),
                        format_pct(metrics["parse_fail_rate"]),
                        format_float(metrics["avg_response_tokens"]),
                        format_float(metrics["avg_rationale_tokens"]),
                    ]
                )
        report.extend(
            [
                "",
                "## Difficulty",
                "",
                markdown_table(
                    [
                        "checkpoint",
                        "bucket",
                        "num_examples",
                        "accuracy",
                        "parse_fail_rate",
                        "avg_response_tokens",
                        "avg_rationale_tokens",
                    ],
                    difficulty_rows,
                ),
            ]
        )

    if error_recovery:
        report.extend(
            [
                "",
                "## Error Recovery",
                "",
                markdown_table(
                    [
                        "num_shared_examples",
                        "wrong_before",
                        "wrong_before_correct_after",
                        "error_recovery_rate",
                        "correct_before_wrong_after",
                        "regression_rate_among_correct_before",
                    ],
                    [
                        [
                            error_recovery["num_shared_examples"],
                            error_recovery["wrong_before"],
                            error_recovery["wrong_before_correct_after"],
                            format_pct(error_recovery["error_recovery_rate"]),
                            error_recovery["correct_before_wrong_after"],
                            format_pct(
                                error_recovery[
                                    "regression_rate_among_correct_before"
                                ]
                            ),
                        ]
                    ],
                ),
            ]
        )

    return "\n".join(report) + "\n"

"""Evaluate cold-start and on-policy checkpoints.

Run from the project root:
python eval/evaluate.py

The script writes JSON/JSONL results, CSV/Markdown tables, and then tries to
generate charts with eval/plot_results.py if matplotlib is available.
"""

import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_utils import (
    ROOT_DIR,
    build_prompt,
    build_report,
    compute_error_recovery,
    ensure_pad_token,
    get_device,
    get_model_input_device,
    get_torch_dtype,
    load_latest_checkpoint,
    parse_response,
    progress,
    read_jsonl,
    summarize_results,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)


EVAL_FILE = ROOT_DIR / "data" / "processed" / "gsm8k_test.jsonl"
FALLBACK_EVAL_FILE = ROOT_DIR / "data" / "cold_start" / "cold_start_gsm8k_train.jsonl"

COLD_START_CHECKPOINT = ROOT_DIR / "outputs" / "cold_start" / "student_cold_start"
LATEST_COLD_START_CHECKPOINT = ROOT_DIR / "outputs" / "cold_start" / "latest_checkpoint.txt"

ON_POLICY_CHECKPOINT = ROOT_DIR / "outputs" / "on_policy" / "student_on_policy"
LATEST_ON_POLICY_CHECKPOINT = ROOT_DIR / "outputs" / "on_policy" / "latest_checkpoint.txt"

OUTPUT_DIR = ROOT_DIR / "outputs" / "eval_results"

CHECKPOINTS = [
    {
        "name": "cold_start",
        "checkpoint": COLD_START_CHECKPOINT,
        "latest_file": LATEST_COLD_START_CHECKPOINT,
    },
    {
        "name": "on_policy",
        "checkpoint": ON_POLICY_CHECKPOINT,
        "latest_file": LATEST_ON_POLICY_CHECKPOINT,
    },
]

# Set LIMIT to a small integer for a quick debug run.
LIMIT = None
SKIP_MISSING_CHECKPOINTS = False

BATCH_SIZE = 1
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 0.2
TOP_P = 0.9
TORCH_DTYPE = "auto"  # auto, float16, bfloat16, float32

RUN_PLOT_SCRIPT = True


def batched(records, batch_size):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def decode_new_tokens(tokenizer, output_ids, prompt_length):
    responses = []
    for ids in output_ids:
        generated_ids = ids[prompt_length:]
        responses.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return responses


def choose_eval_file():
    if Path(EVAL_FILE).exists():
        return Path(EVAL_FILE)
    if Path(FALLBACK_EVAL_FILE).exists():
        return Path(FALLBACK_EVAL_FILE)
    raise FileNotFoundError(
        f"No eval file found. Expected {EVAL_FILE}, or fallback {FALLBACK_EVAL_FILE}."
    )


def resolve_checkpoints():
    resolved = []
    for spec in CHECKPOINTS:
        checkpoint = load_latest_checkpoint(spec["latest_file"], spec["checkpoint"])
        if not checkpoint.exists():
            if SKIP_MISSING_CHECKPOINTS:
                print(f"Skipping missing checkpoint {spec['name']}: {checkpoint}")
                continue
            raise FileNotFoundError(
                f"Checkpoint {spec['name']} not found at {checkpoint}."
            )
        resolved.append({"name": spec["name"], "checkpoint": checkpoint})
    if not resolved:
        raise ValueError("No checkpoints available for evaluation.")
    return resolved


def evaluate_checkpoint(spec, records, device, dtype):
    checkpoint_name = spec["name"]
    checkpoint = spec["checkpoint"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    generation_args = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if DO_SAMPLE:
        generation_args["temperature"] = TEMPERATURE
        generation_args["top_p"] = TOP_P

    results = []
    batches = list(batched(records, BATCH_SIZE))
    for batch in progress(batches, desc=f"eval {checkpoint_name}"):
        prompts = [build_prompt(record) for record in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        input_device = get_model_input_device(model)
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(**encoded, **generation_args)

        responses = decode_new_tokens(tokenizer, output_ids, prompt_length)
        for record, response in zip(batch, responses):
            parsed = parse_response(response, record.get("gold_answer"))
            result = {
                "checkpoint_name": checkpoint_name,
                "checkpoint": str(checkpoint),
                "id": record.get("id"),
                "question": record.get("question"),
                "gold_answer": record.get("gold_answer"),
                "difficulty_level": record.get("difficulty_level"),
                "difficulty_score": record.get("difficulty_score"),
                "model_response": parsed["response"],
                "model_rationale": parsed["rationale"],
                "model_answer": parsed["answer"],
                "model_answer_parsed": parsed["answer_parsed"],
                "model_answer_normalized": parsed["answer_normalized"],
                "correct": parsed["correct"],
                "parse_failed": parsed["parse_failed"],
                "missing_required_sections": parsed["missing_required_sections"],
                "response_tokens": parsed["response_tokens"],
                "rationale_tokens": parsed["rationale_tokens"],
            }
            results.append(result)

    summary = summarize_results(results, checkpoint_name, checkpoint)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, summary


def write_summary_tables(summary, error_recovery):
    overall_rows = []
    for item in summary["checkpoints"]:
        overall_rows.append(
            {
                "checkpoint": item["checkpoint_name"],
                "num_examples": item["num_examples"],
                "accuracy": item["accuracy"],
                "parse_fail_rate": item["parse_fail_rate"],
                "avg_response_tokens": item["avg_response_tokens"],
                "avg_rationale_tokens": item["avg_rationale_tokens"],
            }
        )
    write_csv(
        OUTPUT_DIR / "overall_metrics.csv",
        overall_rows,
        [
            "checkpoint",
            "num_examples",
            "accuracy",
            "parse_fail_rate",
            "avg_response_tokens",
            "avg_rationale_tokens",
        ],
    )

    difficulty_rows = []
    for item in summary["checkpoints"]:
        for level in ["easy", "medium", "hard"]:
            metrics = item.get("difficulty", {}).get(level)
            if metrics is None:
                continue
            difficulty_rows.append(
                {
                    "checkpoint": item["checkpoint_name"],
                    "bucket": level,
                    "num_examples": metrics["num_examples"],
                    "accuracy": metrics["accuracy"],
                    "parse_fail_rate": metrics["parse_fail_rate"],
                    "avg_response_tokens": metrics["avg_response_tokens"],
                    "avg_rationale_tokens": metrics["avg_rationale_tokens"],
                }
            )
    if difficulty_rows:
        write_csv(
            OUTPUT_DIR / "difficulty_metrics.csv",
            difficulty_rows,
            [
                "checkpoint",
                "bucket",
                "num_examples",
                "accuracy",
                "parse_fail_rate",
                "avg_response_tokens",
                "avg_rationale_tokens",
            ],
        )

    if error_recovery:
        write_csv(
            OUTPUT_DIR / "error_recovery.csv",
            [error_recovery],
            [
                "num_shared_examples",
                "wrong_before",
                "wrong_before_correct_after",
                "error_recovery_rate",
                "correct_before_wrong_after",
                "regression_rate_among_correct_before",
            ],
        )

    write_text(OUTPUT_DIR / "report.md", build_report(summary, error_recovery))


def maybe_make_plots():
    if not RUN_PLOT_SCRIPT:
        return
    try:
        subprocess.run(
            [sys.executable, str(ROOT_DIR / "eval" / "plot_results.py")],
            cwd=ROOT_DIR,
            check=True,
        )
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")


def main():
    eval_file = choose_eval_file()
    records = read_jsonl(eval_file)
    if LIMIT is not None:
        records = records[:LIMIT]
    if not records:
        raise ValueError(f"No eval records found in {eval_file}")

    checkpoints = resolve_checkpoints()
    device = get_device()
    dtype = get_torch_dtype(device, TORCH_DTYPE)

    summaries = []
    all_results = {}
    for spec in checkpoints:
        results, summary = evaluate_checkpoint(spec, records, device, dtype)
        results_file = OUTPUT_DIR / f"{spec['name']}_predictions.jsonl"
        summary_file = OUTPUT_DIR / f"{spec['name']}_metrics.json"
        write_jsonl(results_file, results)
        write_json(summary_file, summary)
        summaries.append(summary)
        all_results[spec["name"]] = results
        print(f"Wrote predictions to {results_file}")
        print(f"Wrote metrics to {summary_file}")

    error_recovery = None
    if "cold_start" in all_results and "on_policy" in all_results:
        error_recovery = compute_error_recovery(
            all_results["cold_start"], all_results["on_policy"]
        )
        write_json(OUTPUT_DIR / "error_recovery.json", error_recovery)

    combined_summary = {
        "eval_file": str(eval_file),
        "num_eval_examples": len(records),
        "generation": {
            "do_sample": DO_SAMPLE,
            "temperature": TEMPERATURE if DO_SAMPLE else None,
            "top_p": TOP_P if DO_SAMPLE else None,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "checkpoints": summaries,
        "error_recovery": error_recovery,
    }
    write_json(OUTPUT_DIR / "summary.json", combined_summary)
    write_summary_tables(combined_summary, error_recovery)
    maybe_make_plots()

    print(f"Wrote combined summary to {OUTPUT_DIR / 'summary.json'}")
    print(f"Wrote Markdown report to {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()

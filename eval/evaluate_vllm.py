"""Evaluate cold-start and on-policy checkpoints with vLLM.

Run from the project root:
python eval/evaluate_vllm.py

This writes the same outputs as eval/evaluate.py, but uses vLLM offline batched
generation for much faster evaluation.
"""

import gc
import subprocess
import sys
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

from eval_utils import (
    ROOT_DIR,
    build_prompt,
    build_report,
    compute_error_recovery,
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

# vLLM handles internal micro-batching. Increase until throughput stops
# improving or memory becomes tight.
BATCH_SIZE = 512

MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0

TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 1536
DTYPE = "auto"  # auto, half, bfloat16, float
TRUST_REMOTE_CODE = True

RUN_PLOT_SCRIPT = True


def batched(records, batch_size):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


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


def extract_vllm_texts(outputs):
    texts = []
    for output in outputs:
        if not output.outputs:
            texts.append("")
        else:
            texts.append(output.outputs[0].text.strip())
    return texts


def unload_llm(llm):
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_checkpoint(spec, records):
    checkpoint_name = spec["name"]
    checkpoint = spec["checkpoint"]

    llm = LLM(
        model=str(checkpoint),
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        dtype=DTYPE,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE if DO_SAMPLE else 0.0,
        top_p=TOP_P if DO_SAMPLE else 1.0,
    )

    results = []
    batches = list(batched(records, BATCH_SIZE))
    for batch in progress(batches, desc=f"vllm eval {checkpoint_name}"):
        prompts = [build_prompt(record) for record in batch]
        outputs = llm.generate(prompts, sampling_params)
        responses = extract_vllm_texts(outputs)

        for record, response in zip(batch, responses):
            parsed = parse_response(response, record.get("gold_answer"))
            results.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "checkpoint": str(checkpoint),
                    "generation_backend": "vllm",
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
            )

    summary = summarize_results(results, checkpoint_name, checkpoint)
    summary["generation_backend"] = "vllm"
    unload_llm(llm)
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

    summaries = []
    all_results = {}
    for spec in checkpoints:
        results, summary = evaluate_checkpoint(spec, records)
        results_file = OUTPUT_DIR / f"{spec['name']}_predictions.jsonl"
        summary_file = OUTPUT_DIR / f"{spec['name']}_metrics.json"
        write_jsonl(results_file, results)
        write_json(summary_file, summary)
        summaries.append(summary)
        all_results[spec["name"]] = results
        print(f"Wrote predictions to {results_file}")
        print(f"Wrote metrics to {summary_file}")

    error_recovery = None
    after_checkpoint_names = [
        spec["name"]
        for spec in checkpoints
        if spec["name"] != "cold_start" and spec["name"] in all_results
    ]
    if "cold_start" in all_results and after_checkpoint_names:
        after_name = after_checkpoint_names[0]
        error_recovery = compute_error_recovery(
            all_results["cold_start"], all_results[after_name]
        )
        write_json(OUTPUT_DIR / "error_recovery.json", error_recovery)

    combined_summary = {
        "eval_file": str(eval_file),
        "num_eval_examples": len(records),
        "generation_backend": "vllm",
        "generation": {
            "do_sample": DO_SAMPLE,
            "temperature": TEMPERATURE if DO_SAMPLE else None,
            "top_p": TOP_P if DO_SAMPLE else None,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": BATCH_SIZE,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "max_model_len": MAX_MODEL_LEN,
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

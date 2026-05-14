"""Generate on-policy student rollouts with vLLM offline batching.

Run from the project root:
python stage_B/generate_student_rollouts_vllm.py

This writes the same output schema as generate_student_rollouts.py, so
train_on_policy_logits_kd.py does not change.
"""

from pathlib import Path

from vllm import LLM, SamplingParams

from train_utils import (
    ROOT_DIR,
    append_jsonl,
    build_eval_metric_schema,
    build_prompt,
    load_completed_ids,
    load_latest_checkpoint,
    parse_student_response,
    progress,
    read_jsonl,
    summarize_generation_records,
    write_json,
)


SOURCE_FILE = ROOT_DIR / "data" / "cold_start" / "cold_start_gsm8k_train.jsonl"
COLD_START_CHECKPOINT = ROOT_DIR / "outputs" / "cold_start" / "student_cold_start"
LATEST_COLD_START_CHECKPOINT = ROOT_DIR / "outputs" / "cold_start" / "latest_checkpoint.txt"

OUTPUT_FILE = ROOT_DIR / "outputs" / "on_policy" / "rollouts" / "student_rollouts_gsm8k_train.jsonl"
STATS_FILE = ROOT_DIR / "outputs" / "on_policy" / "rollouts" / "student_rollout_stats.json"

# Set LIMIT to a small integer for a quick debug run.
LIMIT = None
RESUME = False

# vLLM handles internal micro-batching. This controls how many prompts are
# submitted per generate call.
BATCH_SIZE = 128

MAX_NEW_TOKENS = 256

# Rollouts should be sampled from the current student policy.
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9

TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 1536
DTYPE = "auto"  # auto, half, bfloat16, float
TRUST_REMOTE_CODE = True


def batched(records, batch_size):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def extract_vllm_texts(outputs):
    texts = []
    for output in outputs:
        if not output.outputs:
            texts.append("")
        else:
            texts.append(output.outputs[0].text.strip())
    return texts


def main():
    checkpoint = load_latest_checkpoint(
        LATEST_COLD_START_CHECKPOINT, COLD_START_CHECKPOINT
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Cold-start checkpoint not found at {checkpoint}. "
            "Run Stage A before generating Stage B rollouts."
        )

    records = read_jsonl(SOURCE_FILE)
    if LIMIT is not None:
        records = records[:LIMIT]

    if RESUME:
        completed_ids = load_completed_ids(OUTPUT_FILE)
        records = [record for record in records if str(record["id"]) not in completed_ids]
    elif Path(OUTPUT_FILE).exists():
        Path(OUTPUT_FILE).unlink()

    if not records:
        raise ValueError(f"No source records found in {SOURCE_FILE}")

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

    written = []
    batches = list(batched(records, BATCH_SIZE))
    for batch in progress(batches, desc="vllm student rollouts"):
        prompts = [build_prompt(record) for record in batch]
        outputs = llm.generate(prompts, sampling_params)
        responses = extract_vllm_texts(outputs)

        for record, response in zip(batch, responses):
            rollout = dict(record)
            rollout["student_checkpoint"] = str(checkpoint)
            rollout["generation_backend"] = "vllm"
            rollout["student_response"] = response
            rollout.update(parse_student_response(response, record.get("gold_answer")))
            append_jsonl(OUTPUT_FILE, rollout)
            written.append(rollout)

    stats = summarize_generation_records(written)
    stats.update(
        {
            "stage": "B_rollout",
            "source_file": str(SOURCE_FILE),
            "student_checkpoint": str(checkpoint),
            "rollout_file": str(OUTPUT_FILE),
            "generation_backend": "vllm",
            "generation": {
                "do_sample": DO_SAMPLE,
                "temperature": TEMPERATURE if DO_SAMPLE else None,
                "top_p": TOP_P if DO_SAMPLE else None,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            "eval_metric_schema": build_eval_metric_schema(),
        }
    )
    write_json(STATS_FILE, stats)

    print(f"Wrote {len(written)} student rollouts to {OUTPUT_FILE}")
    print(f"Wrote rollout stats to {STATS_FILE}")


if __name__ == "__main__":
    main()

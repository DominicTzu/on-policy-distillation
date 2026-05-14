"""Generate on-policy student rollouts for Stage B.

Run from the project root:
python stage_B/generate_student_rollouts.py

This starts from the Stage A cold-start checkpoint and saves the student
trajectories that Stage B trains on.
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_utils import (
    ROOT_DIR,
    append_jsonl,
    build_eval_metric_schema,
    build_prompt,
    ensure_pad_token,
    get_device,
    get_model_input_device,
    get_torch_dtype,
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

BATCH_SIZE = 1
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 256

# On-policy training should use trajectories sampled from the student policy.
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9

TORCH_DTYPE = "auto"  # auto, float16, bfloat16, float32


def batched(records, batch_size):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def decode_new_tokens(tokenizer, output_ids, prompt_length):
    responses = []
    for ids in output_ids:
        generated_ids = ids[prompt_length:]
        responses.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return responses


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

    device = get_device()
    dtype = get_torch_dtype(device, TORCH_DTYPE)

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

    written = []
    batches = list(batched(records, BATCH_SIZE))
    for batch in progress(batches, desc="student rollouts"):
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
            rollout = dict(record)
            rollout["student_checkpoint"] = str(checkpoint)
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

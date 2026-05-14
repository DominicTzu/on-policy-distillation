"""Compress teacher long rationales with difficulty-aware instructions.

Run from the project root:
python data/compress_teacher_rationales.py
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from answer_utils import extract_answer, split_rationale_and_answer
from data_utils import append_jsonl, iter_jsonl, read_jsonl
from generation_utils import decode_new_tokens, get_model_input_device, progress
from prompts import build_compression_messages, build_compression_prompt


BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "cold_start" / "teacher_long_gsm8k_train.difficulty.jsonl"
OUTPUT_FILE = BASE_DIR / "cold_start" / "teacher_compressed_gsm8k_train.jsonl"

TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 1
LIMIT = None
RESUME = False

MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 0.2
TOP_P = 0.9

DEVICE_MAP = "auto"
TORCH_DTYPE = "auto"  # auto, float16, bfloat16, float32


def get_torch_dtype():
    if TORCH_DTYPE == "auto":
        return "auto"
    if TORCH_DTYPE == "float16":
        return torch.float16
    if TORCH_DTYPE == "bfloat16":
        return torch.bfloat16
    if TORCH_DTYPE == "float32":
        return torch.float32
    raise ValueError(f"Unknown TORCH_DTYPE: {TORCH_DTYPE}")


def batched(records, batch_size):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def load_completed_ids(path):
    if not Path(path).exists():
        return set()
    completed_ids = set()
    for record in iter_jsonl(path):
        if "id" in record:
            completed_ids.add(str(record["id"]))
    return completed_ids


def build_model_inputs(tokenizer, records):
    prompts = []
    for record in records:
        messages = build_compression_messages(record)
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = build_compression_prompt(record)
        prompts.append(prompt)
    return prompts


def main():
    records = read_jsonl(INPUT_FILE)
    if LIMIT is not None:
        records = records[:LIMIT]

    if RESUME:
        completed_ids = load_completed_ids(OUTPUT_FILE)
        records = [r for r in records if str(r["id"]) not in completed_ids]

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=get_torch_dtype(),
        device_map=DEVICE_MAP,
        trust_remote_code=True,
    )
    model.eval()

    if Path(OUTPUT_FILE).exists() and not RESUME:
        Path(OUTPUT_FILE).unlink()

    generation_args = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if DO_SAMPLE:
        generation_args["temperature"] = TEMPERATURE
        generation_args["top_p"] = TOP_P

    batches = list(batched(records, BATCH_SIZE))
    for batch in progress(batches, desc="teacher compression"):
        prompts = build_model_inputs(tokenizer, batch)

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_device = get_model_input_device(model)
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(**encoded, **generation_args)

        responses = decode_new_tokens(tokenizer, output_ids, prompt_length)
        for record, response in zip(batch, responses):
            rationale, answer = split_rationale_and_answer(response)
            compressed = dict(record)
            compressed["teacher_model"] = TEACHER_MODEL
            compressed["teacher_compressed_response"] = response
            compressed["teacher_compressed_rationale"] = rationale
            compressed["teacher_compressed_answer"] = answer
            compressed["teacher_compressed_answer_parsed"] = extract_answer(response)
            append_jsonl(OUTPUT_FILE, compressed)

    print(f"Wrote teacher compressed rationales to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

"""Generate teacher long-CoT outputs for Stage A.

Run from the project root:
python data/generate_teacher_cold_start.py

This is the first teacher-generation step in the README pipeline:
question -> teacher long CoT + answer.
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from answer_utils import extract_answer, split_long_rationale_and_answer
from data_utils import append_jsonl, iter_jsonl, read_jsonl
from generation_utils import decode_new_tokens, get_model_input_device, progress
from prompts import build_long_cot_messages, build_long_cot_prompt


BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "processed" / "gsm8k_train.jsonl"
OUTPUT_FILE = BASE_DIR / "cold_start" / "teacher_long_gsm8k_train.jsonl"

TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 1
LIMIT = None
RESUME = False

MAX_NEW_TOKENS = 512
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


def build_model_inputs(tokenizer, questions):
    prompts = []
    for question in questions:
        messages = build_long_cot_messages(question)
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = build_long_cot_prompt(question)
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
    for batch in progress(batches, desc="teacher long cot"):
        questions = [str(record["question"]) for record in batch]
        prompts = build_model_inputs(tokenizer, questions)

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
            rationale, answer = split_long_rationale_and_answer(response)
            generated = dict(record)
            generated["teacher_model"] = TEACHER_MODEL
            generated["teacher_long_response"] = response
            generated["teacher_long_rationale"] = rationale
            generated["teacher_long_answer"] = answer
            generated["teacher_long_answer_parsed"] = extract_answer(response)
            append_jsonl(OUTPUT_FILE, generated)

    print(f"Wrote teacher long-CoT generations to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

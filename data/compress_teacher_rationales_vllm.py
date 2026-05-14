"""Compress teacher long rationales with vLLM offline batching.

Run from the project root:
python data/compress_teacher_rationales_vllm.py

This writes the same output schema as compress_teacher_rationales.py.
"""

from pathlib import Path

from vllm import LLM, SamplingParams

from answer_utils import extract_answer, split_rationale_and_answer
from data_utils import append_jsonl, read_jsonl
from prompts import build_compression_messages, build_compression_prompt
from vllm_generation_utils import (
    batched,
    build_chat_prompt,
    extract_vllm_texts,
    load_completed_ids,
    progress,
)


BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "cold_start" / "teacher_long_gsm8k_train.difficulty.jsonl"
OUTPUT_FILE = BASE_DIR / "cold_start" / "teacher_compressed_gsm8k_train.jsonl"

TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LIMIT = None
RESUME = False

BATCH_SIZE = 128

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = 1.0

TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 3072
DTYPE = "auto"  # auto, half, bfloat16, float
TRUST_REMOTE_CODE = True


def build_prompts(tokenizer, records):
    prompts = []
    for record in records:
        messages = build_compression_messages(record)
        fallback_prompt = build_compression_prompt(record)
        prompts.append(build_chat_prompt(tokenizer, messages, fallback_prompt))
    return prompts


def main():
    records = read_jsonl(INPUT_FILE)
    if LIMIT is not None:
        records = records[:LIMIT]

    if RESUME:
        completed_ids = load_completed_ids(OUTPUT_FILE)
        records = [r for r in records if str(r["id"]) not in completed_ids]
    elif Path(OUTPUT_FILE).exists():
        Path(OUTPUT_FILE).unlink()

    if not records:
        raise ValueError(f"No input records found in {INPUT_FILE}")

    llm = LLM(
        model=TEACHER_MODEL,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        dtype=DTYPE,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    batches = list(batched(records, BATCH_SIZE))
    for batch in progress(batches, desc="vllm teacher compression"):
        prompts = build_prompts(tokenizer, batch)
        outputs = llm.generate(prompts, sampling_params)
        responses = extract_vllm_texts(outputs)

        for record, response in zip(batch, responses):
            rationale, answer = split_rationale_and_answer(response)
            compressed = dict(record)
            compressed["teacher_model"] = TEACHER_MODEL
            compressed["generation_backend"] = "vllm"
            compressed["teacher_compressed_response"] = response
            compressed["teacher_compressed_rationale"] = rationale
            compressed["teacher_compressed_answer"] = answer
            compressed["teacher_compressed_answer_parsed"] = extract_answer(response)
            append_jsonl(OUTPUT_FILE, compressed)

    print(f"Wrote teacher compressed rationales to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

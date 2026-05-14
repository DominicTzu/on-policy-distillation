"""Generate teacher long-CoT outputs with vLLM offline batching.

Run from the project root:
python data/generate_teacher_cold_start_vllm.py

This writes the same output schema as generate_teacher_cold_start.py, so the
downstream filtering and difficulty scripts do not change.
"""

from pathlib import Path

from vllm import LLM, SamplingParams

from answer_utils import extract_answer, split_long_rationale_and_answer
from data_utils import append_jsonl, read_jsonl
from prompts import build_long_cot_messages, build_long_cot_prompt
from vllm_generation_utils import (
    batched,
    build_chat_prompt,
    extract_vllm_texts,
    load_completed_ids,
    progress,
)


BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "processed" / "gsm8k_train.jsonl"
OUTPUT_FILE = BASE_DIR / "cold_start" / "teacher_long_gsm8k_train.jsonl"

TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LIMIT = None
RESUME = False

# vLLM handles internal micro-batching; this is the number of prompts submitted
# per generate call. Increase until throughput stops improving or memory is tight.
BATCH_SIZE = 128

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
TOP_P = 1.0

TENSOR_PARALLEL_SIZE = 2
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 2048
DTYPE = "auto"  # auto, half, bfloat16, float
TRUST_REMOTE_CODE = True


def build_prompts(tokenizer, records):
    prompts = []
    for record in records:
        question = str(record["question"])
        messages = build_long_cot_messages(question)
        fallback_prompt = build_long_cot_prompt(question)
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
    for batch in progress(batches, desc="vllm teacher long cot"):
        prompts = build_prompts(tokenizer, batch)
        outputs = llm.generate(prompts, sampling_params)
        responses = extract_vllm_texts(outputs)

        for record, response in zip(batch, responses):
            rationale, answer = split_long_rationale_and_answer(response)
            generated = dict(record)
            generated["teacher_model"] = TEACHER_MODEL
            generated["generation_backend"] = "vllm"
            generated["teacher_long_response"] = response
            generated["teacher_long_rationale"] = rationale
            generated["teacher_long_answer"] = answer
            generated["teacher_long_answer_parsed"] = extract_answer(response)
            append_jsonl(OUTPUT_FILE, generated)

    print(f"Wrote teacher long-CoT generations to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

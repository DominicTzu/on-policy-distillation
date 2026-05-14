"""Helpers for vLLM offline batched generation."""

from pathlib import Path

from data_utils import iter_jsonl


def progress(items, desc=None):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc)
    except ImportError:
        return items


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


def build_chat_prompt(tokenizer, messages, fallback_prompt):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return fallback_prompt


def extract_vllm_texts(outputs):
    texts = []
    for output in outputs:
        if not output.outputs:
            texts.append("")
        else:
            texts.append(output.outputs[0].text.strip())
    return texts

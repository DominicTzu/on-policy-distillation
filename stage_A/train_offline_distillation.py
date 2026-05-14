"""Stage A offline distillation.

Run from the project root:
python stage_A/train_offline_distillation.py

This trains the student on filtered teacher-compressed rationales and writes
the cold-start checkpoint needed by Stage B.
"""

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


ROOT_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = ROOT_DIR / "data" / "cold_start" / "cold_start_gsm8k_train.jsonl"
OUTPUT_DIR = ROOT_DIR / "outputs" / "cold_start" / "student_cold_start"
STAGE_B_MANIFEST = ROOT_DIR / "outputs" / "cold_start" / "stage_b_manifest.json"
LATEST_CHECKPOINT_FILE = ROOT_DIR / "outputs" / "cold_start" / "latest_checkpoint.txt"

STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Set LIMIT to a small integer for a quick debug run.
LIMIT = None

MAX_LENGTH = 1024
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
LOG_EVERY_STEPS = 10

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

USE_BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
USE_FP16 = DEVICE == "cuda" and not USE_BF16


def read_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_prompt(record):
    return "### Question\n" + str(record["question"]).strip() + "\n\n"


def build_target(record):
    if record.get("target_response"):
        return str(record["target_response"]).strip()

    return (
        "### Rationale\n"
        + str(record["teacher_compressed_rationale"]).strip()
        + "\n\n### Answer\n"
        + str(record["teacher_compressed_answer"]).strip()
    )


class ColdStartDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        prompt = build_prompt(record)
        target = build_target(record)
        full_text = prompt + target + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": full_ids,
            "labels": labels,
            "id": record.get("id", str(idx)),
        }


def collate_batch(batch, tokenizer):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def save_stage_b_manifest(records, train_steps, avg_loss):
    manifest = {
        "stage": "A",
        "student_base_model": STUDENT_MODEL,
        "cold_start_checkpoint": str(OUTPUT_DIR),
        "tokenizer_path": str(OUTPUT_DIR),
        "cold_start_train_file": str(TRAIN_FILE),
        "num_train_examples": len(records),
        "train_steps": train_steps,
        "avg_train_loss": avg_loss,
        "prompt_template": "### Question\n{question}\n\n",
        "response_format": "### Rationale\n{rationale}\n\n### Answer\n{answer}",
        "stage_b_expected_input": {
            "checkpoint": str(OUTPUT_DIR),
            "rollout_source_file": str(TRAIN_FILE),
            "student_response_fields": [
                "student_response",
                "student_rationale",
                "student_answer",
                "student_correct",
            ],
        },
    }
    write_json(STAGE_B_MANIFEST, manifest)
    LATEST_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    LATEST_CHECKPOINT_FILE.write_text(str(OUTPUT_DIR) + "\n", encoding="utf-8")


def main():
    records = read_jsonl(TRAIN_FILE)
    if LIMIT is not None:
        records = records[:LIMIT]

    if not records:
        raise ValueError(f"No training records found in {TRAIN_FILE}")

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32
    if USE_BF16:
        dtype = torch.bfloat16
    if USE_FP16:
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(DEVICE)
    model.train()

    dataset = ColdStartDataset(records, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )

    total_update_steps = math.ceil(
        len(dataloader) * NUM_EPOCHS / GRADIENT_ACCUMULATION_STEPS
    )
    warmup_steps = int(total_update_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    losses = []
    update_step = 0
    start_time = time.time()
    optimizer.zero_grad()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with torch.autocast(
                device_type=DEVICE,
                dtype=torch.bfloat16 if USE_BF16 else torch.float16,
                enabled=USE_BF16 or USE_FP16,
            ):
                outputs = model(**batch)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

            if USE_FP16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_update = (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            is_last_batch = batch_idx + 1 == len(dataloader)

            if should_update or is_last_batch:
                if USE_FP16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                update_step += 1
                step_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
                losses.append(step_loss)

                if update_step % LOG_EVERY_STEPS == 0 or update_step == 1:
                    avg_loss = sum(losses[-LOG_EVERY_STEPS:]) / min(
                        len(losses), LOG_EVERY_STEPS
                    )
                    print(
                        f"epoch={epoch + 1} step={update_step}/{total_update_steps} "
                        f"loss={avg_loss:.4f}"
                    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    avg_loss = sum(losses) / len(losses) if losses else None
    train_state = {
        "stage": "A",
        "train_file": str(TRAIN_FILE),
        "output_dir": str(OUTPUT_DIR),
        "student_base_model": STUDENT_MODEL,
        "num_train_examples": len(records),
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "train_steps": update_step,
        "avg_train_loss": avg_loss,
        "elapsed_seconds": time.time() - start_time,
    }
    write_json(OUTPUT_DIR / "stage_a_train_state.json", train_state)
    save_stage_b_manifest(records, update_step, avg_loss)

    print(f"Saved cold-start checkpoint to {OUTPUT_DIR}")
    print(f"Saved Stage B manifest to {STAGE_B_MANIFEST}")


if __name__ == "__main__":
    main()

"""Stage B on-policy logits-level CoT distillation.

Run from the project root:
python stage_B/train_on_policy_logits_kd.py

This trains the cold-start student on its own rollout trajectories by matching
the teacher next-token distributions on the same prompt + student_response
sequences.
"""

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from logits_kd import distillation_losses
from train_utils import (
    PROMPT_TEMPLATE,
    RESPONSE_TEMPLATE,
    ROOT_DIR,
    build_eval_metric_schema,
    build_prompt,
    ensure_pad_token,
    get_device,
    get_model_input_device,
    get_torch_dtype,
    load_latest_checkpoint,
    read_jsonl,
    summarize_generation_records,
    write_json,
)


ROLLOUT_FILE = (
    ROOT_DIR
    / "outputs"
    / "on_policy"
    / "rollouts"
    / "student_rollouts_gsm8k_train.jsonl"
)
COLD_START_CHECKPOINT = ROOT_DIR / "outputs" / "cold_start" / "student_cold_start"
LATEST_COLD_START_CHECKPOINT = ROOT_DIR / "outputs" / "cold_start" / "latest_checkpoint.txt"

OUTPUT_DIR = ROOT_DIR / "outputs" / "on_policy" / "student_on_policy"
STAGE_B_MANIFEST = ROOT_DIR / "outputs" / "on_policy" / "stage_b_manifest.json"
LATEST_ON_POLICY_CHECKPOINT = ROOT_DIR / "outputs" / "on_policy" / "latest_checkpoint.txt"

TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Set LIMIT to a small integer for a quick debug run.
LIMIT = None

MAX_LENGTH = 1024
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
LOG_EVERY_STEPS = 10

TEMPERATURE = 2.0
ALPHA_CE = 0.5
BETA_KD = 1.0

STUDENT_TORCH_DTYPE = "auto"  # auto, float16, bfloat16, float32
TEACHER_TORCH_DTYPE = "auto"


class OnPolicyRolloutDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.examples = []
        self.skipped_no_response_tokens = 0

        for idx, record in enumerate(records):
            example = self.build_example(record, tokenizer, max_length, idx)
            if example is None:
                self.skipped_no_response_tokens += 1
            else:
                self.examples.append(example)

    def build_example(self, record, tokenizer, max_length, idx):
        prompt = build_prompt(record)
        response = str(record.get("student_response", "")).strip()
        full_text = prompt + response + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        response_token_count = sum(1 for label in labels if label != -100)
        if response_token_count == 0:
            return None

        return {
            "input_ids": full_ids,
            "labels": labels,
            "id": record.get("id", str(idx)),
            "response_token_count": response_token_count,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


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


def load_student(checkpoint, tokenizer, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return model


def load_teacher(dtype, device):
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def save_stage_b_manifest(
    cold_start_checkpoint,
    rollout_metrics,
    train_state,
):
    manifest = {
        "stage": "B",
        "method": "on_policy_logits_level_cot_distillation",
        "cold_start_checkpoint": str(cold_start_checkpoint),
        "on_policy_checkpoint": str(OUTPUT_DIR),
        "teacher_model": TEACHER_MODEL,
        "rollout_file": str(ROLLOUT_FILE),
        "prompt_template": PROMPT_TEMPLATE,
        "response_format": RESPONSE_TEMPLATE,
        "loss": {
            "temperature": TEMPERATURE,
            "alpha_ce": ALPHA_CE,
            "beta_kd": BETA_KD,
            "formula": "alpha_ce * CE + beta_kd * T^2 * KL(teacher || student)",
        },
        "rollout_metrics": rollout_metrics,
        "train_state_file": str(OUTPUT_DIR / "stage_b_train_state.json"),
        "eval_metric_schema": build_eval_metric_schema(),
        "eval_checkpoints": {
            "cold_start": str(cold_start_checkpoint),
            "on_policy": str(OUTPUT_DIR),
        },
        "train_state": train_state,
    }
    write_json(STAGE_B_MANIFEST, manifest)
    LATEST_ON_POLICY_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    LATEST_ON_POLICY_CHECKPOINT.write_text(str(OUTPUT_DIR) + "\n", encoding="utf-8")


def main():
    cold_start_checkpoint = load_latest_checkpoint(
        LATEST_COLD_START_CHECKPOINT, COLD_START_CHECKPOINT
    )
    if not cold_start_checkpoint.exists():
        raise FileNotFoundError(
            f"Cold-start checkpoint not found at {cold_start_checkpoint}. "
            "Run Stage A before Stage B training."
        )

    records = read_jsonl(ROLLOUT_FILE)
    if LIMIT is not None:
        records = records[:LIMIT]
    if not records:
        raise ValueError(
            f"No rollout records found in {ROLLOUT_FILE}. "
            "Run stage_B/generate_student_rollouts.py first."
        )

    device = get_device()
    student_dtype = get_torch_dtype(device, STUDENT_TORCH_DTYPE)
    teacher_dtype = get_torch_dtype(device, TEACHER_TORCH_DTYPE)

    tokenizer = AutoTokenizer.from_pretrained(
        cold_start_checkpoint, trust_remote_code=True
    )
    ensure_pad_token(tokenizer)

    dataset = OnPolicyRolloutDataset(records, tokenizer, MAX_LENGTH)
    if len(dataset) == 0:
        raise ValueError("No rollout examples have response tokens after truncation.")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )

    student = load_student(cold_start_checkpoint, tokenizer, student_dtype, device)
    teacher = load_teacher(teacher_dtype, device)
    student.train()

    total_update_steps = math.ceil(
        len(dataloader) * NUM_EPOCHS / GRADIENT_ACCUMULATION_STEPS
    )
    warmup_steps = int(total_update_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    use_bf16 = device == "cuda" and student_dtype == torch.bfloat16
    use_fp16 = device == "cuda" and student_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    total_losses = []
    ce_losses = []
    kd_losses = []
    update_step = 0
    start_time = time.time()
    optimizer.zero_grad()

    student_input_device = get_model_input_device(student)
    teacher_input_device = get_model_input_device(teacher)

    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(dataloader):
            window_start = (batch_idx // GRADIENT_ACCUMULATION_STEPS) * (
                GRADIENT_ACCUMULATION_STEPS
            )
            window_end = min(
                window_start + GRADIENT_ACCUMULATION_STEPS, len(dataloader)
            )
            accumulation_size = window_end - window_start

            student_batch = {
                key: value.to(student_input_device)
                for key, value in batch.items()
                if key != "labels"
            }
            teacher_batch = {
                key: value.to(teacher_input_device)
                for key, value in batch.items()
                if key != "labels"
            }
            labels = batch["labels"].to(student_input_device)

            with torch.no_grad():
                teacher_outputs = teacher(**teacher_batch)

            with torch.autocast(
                device_type=device,
                dtype=student_dtype,
                enabled=use_bf16 or use_fp16,
            ):
                student_outputs = student(**student_batch)
                teacher_logits = teacher_outputs.logits.to(student_outputs.logits.device)
                ce_loss, kd_loss = distillation_losses(
                    student_outputs.logits,
                    teacher_logits,
                    labels,
                    TEMPERATURE,
                )
                total_loss = ALPHA_CE * ce_loss + BETA_KD * kd_loss
                loss_for_backward = total_loss / accumulation_size

            if use_fp16:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            total_losses.append(total_loss.item())
            ce_losses.append(ce_loss.item())
            kd_losses.append(kd_loss.item())

            should_update = (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            is_last_batch = batch_idx + 1 == len(dataloader)

            if should_update or is_last_batch:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                update_step += 1

                if update_step % LOG_EVERY_STEPS == 0 or update_step == 1:
                    log_count = min(len(total_losses), LOG_EVERY_STEPS)
                    avg_total = sum(total_losses[-log_count:]) / log_count
                    avg_ce = sum(ce_losses[-log_count:]) / log_count
                    avg_kd = sum(kd_losses[-log_count:]) / log_count
                    print(
                        f"epoch={epoch + 1} step={update_step}/{total_update_steps} "
                        f"loss={avg_total:.4f} ce={avg_ce:.4f} kd={avg_kd:.4f}"
                    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    rollout_metrics = summarize_generation_records(records)
    train_state = {
        "stage": "B",
        "method": "on_policy_logits_level_cot_distillation",
        "rollout_file": str(ROLLOUT_FILE),
        "output_dir": str(OUTPUT_DIR),
        "cold_start_checkpoint": str(cold_start_checkpoint),
        "teacher_model": TEACHER_MODEL,
        "num_rollouts": len(records),
        "num_train_examples": len(dataset),
        "skipped_no_response_tokens": dataset.skipped_no_response_tokens,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "temperature": TEMPERATURE,
        "alpha_ce": ALPHA_CE,
        "beta_kd": BETA_KD,
        "train_steps": update_step,
        "avg_total_loss": sum(total_losses) / len(total_losses)
        if total_losses
        else None,
        "avg_ce_loss": sum(ce_losses) / len(ce_losses) if ce_losses else None,
        "avg_kd_loss": sum(kd_losses) / len(kd_losses) if kd_losses else None,
        "elapsed_seconds": time.time() - start_time,
        "rollout_metrics": rollout_metrics,
        "eval_metric_schema": build_eval_metric_schema(),
    }
    write_json(OUTPUT_DIR / "stage_b_train_state.json", train_state)
    save_stage_b_manifest(cold_start_checkpoint, rollout_metrics, train_state)

    print(f"Saved on-policy checkpoint to {OUTPUT_DIR}")
    print(f"Saved Stage B manifest to {STAGE_B_MANIFEST}")


if __name__ == "__main__":
    main()

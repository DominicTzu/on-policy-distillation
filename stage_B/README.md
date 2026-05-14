# Stage B: On-Policy Logits-Level Distillation

Run these commands from the project root after Stage A has produced:

```text
outputs/cold_start/student_cold_start/
outputs/cold_start/latest_checkpoint.txt
data/cold_start/cold_start_gsm8k_train.jsonl
```

## 1. Generate Student Rollouts

Recommended vLLM version:

```bash
python stage_B/generate_student_rollouts_vllm.py
```

Baseline Transformers version:

```bash
python stage_B/generate_student_rollouts.py
```

Input:

```text
data/cold_start/cold_start_gsm8k_train.jsonl
outputs/cold_start/latest_checkpoint.txt
```

Writes:

```text
outputs/on_policy/rollouts/student_rollouts_gsm8k_train.jsonl
outputs/on_policy/rollouts/student_rollout_stats.json
```

Each rollout record keeps the README fields:

```json
{
  "id": "...",
  "question": "...",
  "gold_answer": "...",
  "student_response": "### Rationale\n...\n\n### Answer\n...",
  "student_rationale": "...",
  "student_answer": "...",
  "student_correct": true
}
```

It also stores eval-ready fields:

```text
student_answer_parsed
student_answer_normalized
student_parse_failed
student_response_tokens
student_rationale_tokens
difficulty_level
```

`student_rollout_stats.json` reports accuracy, parse fail rate, average response
tokens, average rationale tokens, and difficulty-bucket metrics when the source
records include `difficulty_level`.

## 2. Train With Teacher Logits

```bash
python stage_B/train_on_policy_logits_kd.py
```

Input:

```text
outputs/on_policy/rollouts/student_rollouts_gsm8k_train.jsonl
outputs/cold_start/latest_checkpoint.txt
```

Output:

```text
outputs/on_policy/student_on_policy/
outputs/on_policy/student_on_policy/stage_b_train_state.json
outputs/on_policy/stage_b_manifest.json
outputs/on_policy/latest_checkpoint.txt
```

For each rollout, the script constructs:

```text
### Question
{question}

{student_response}
```

Both teacher and student run on exactly this sequence. Prompt tokens are masked,
and CE/KD losses are computed only on response-token prediction positions.

The training loss is:

```text
loss = alpha_ce * CE + beta_kd * T^2 * KL(teacher || student)
```

Change hyperparameters by editing constants at the top of
`stage_B/generate_student_rollouts_vllm.py`,
`stage_B/generate_student_rollouts.py`, and
`stage_B/train_on_policy_logits_kd.py`.

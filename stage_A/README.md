# Stage A: Offline Distillation

Run from the project root:

```bash
python stage_A/train_offline_distillation.py
```

Input:

```text
data/cold_start/cold_start_gsm8k_train.jsonl
```

Output:

```text
outputs/cold_start/student_cold_start/
outputs/cold_start/student_cold_start/stage_a_train_state.json
outputs/cold_start/stage_b_manifest.json
outputs/cold_start/latest_checkpoint.txt
```

The script trains `Qwen/Qwen2.5-1.5B-Instruct` with response-only supervised
learning. Prompt tokens are masked with `-100`, so loss is only computed on:

```text
### Rationale
{teacher_compressed_rationale}

### Answer
{teacher_compressed_answer}
```

Change hyperparameters by editing constants at the top of
`stage_A/train_offline_distillation.py`.

# Ablation Runs

Run from the project root.

## Stage B Ablation

This entrypoint reuses the main Stage B implementation but writes outputs under
`ablation/output/`.

```bash
CUDA_VISIBLE_DEVICES=0,1 python ablation/stage_B/train_on_policy_logits_kd.py
```

Default ablation settings:

```text
ALPHA_CE = 1.0
BETA_KD = 0.2
TEMPERATURE = 2.0
EXCLUDE_EOS_FROM_KD = True
TRAIN_ONLY_CORRECT_ROLLOUTS = True
TRAIN_ONLY_FORMAT_VALID_ROLLOUTS = True
MAX_ROLLOUT_RESPONSE_TOKENS = 128
MAX_ROLLOUT_RATIONALE_TOKENS = 96
```

Input rollout file:

```text
outputs/on_policy/rollouts/student_rollouts_gsm8k_train.jsonl
```

Outputs:

```text
ablation/output/on_policy/student_on_policy/
ablation/output/on_policy/student_on_policy/stage_b_train_state.json
ablation/output/on_policy/stage_b_manifest.json
ablation/output/on_policy/latest_checkpoint.txt
```

## Evaluation

Evaluate the cold-start checkpoint against the ablation on-policy checkpoint:

```bash
python ablation/eval/evaluate_vllm.py
```

This uses the same metrics as the main eval script and does not generate plots.

Outputs:

```text
ablation/output/eval_results/cold_start_predictions.jsonl
ablation/output/eval_results/ablation_on_policy_predictions.jsonl
ablation/output/eval_results/cold_start_metrics.json
ablation/output/eval_results/ablation_on_policy_metrics.json
ablation/output/eval_results/summary.json
ablation/output/eval_results/report.md
ablation/output/eval_results/overall_metrics.csv
ablation/output/eval_results/difficulty_metrics.csv
ablation/output/eval_results/error_recovery.json
ablation/output/eval_results/error_recovery.csv
```

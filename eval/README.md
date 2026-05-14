# Evaluation

Run from the project root after Stage A and Stage B have produced checkpoints:

```bash
python eval/evaluate.py
```

By default, the script evaluates:

```text
outputs/cold_start/latest_checkpoint.txt
outputs/on_policy/latest_checkpoint.txt
```

It uses `data/processed/gsm8k_test.jsonl` when available. If that file does not
exist, it falls back to:

```text
data/cold_start/cold_start_gsm8k_train.jsonl
```

Change `LIMIT` at the top of `eval/evaluate.py` for a quick debug run.

## Outputs

Per-checkpoint predictions:

```text
outputs/eval_results/cold_start_predictions.jsonl
outputs/eval_results/on_policy_predictions.jsonl
```

Per-checkpoint metrics:

```text
outputs/eval_results/cold_start_metrics.json
outputs/eval_results/on_policy_metrics.json
```

Combined metrics and reports:

```text
outputs/eval_results/summary.json
outputs/eval_results/report.md
outputs/eval_results/overall_metrics.csv
outputs/eval_results/difficulty_metrics.csv
outputs/eval_results/error_recovery.json
outputs/eval_results/error_recovery.csv
```

Charts are written when `matplotlib` is installed:

```text
outputs/eval_results/plots/accuracy.png
outputs/eval_results/plots/parse_fail_rate.png
outputs/eval_results/plots/generated_lengths.png
outputs/eval_results/plots/difficulty_accuracy.png
outputs/eval_results/plots/difficulty_rationale_length.png
outputs/eval_results/plots/error_recovery.png
```

You can regenerate charts from an existing summary with:

```bash
python eval/plot_results.py
```

# Data Pipeline

Run these commands from the project root.

## 1. Prepare GSM8K

```bash
python data/prepare_data.py
```

Writes:

```text
data/raw/gsm8k_train.jsonl
data/processed/gsm8k_train.jsonl
```

For a debug run, edit `LIMIT` in `data/prepare_data.py`.

## 2. Generate Teacher Long CoT

Recommended vLLM version:

```bash
python data/generate_teacher_cold_start_vllm.py
```

Baseline Transformers version:

```bash
python data/generate_teacher_cold_start.py
```

Writes:

```text
data/cold_start/teacher_long_gsm8k_train.jsonl
```

## 3. Filter Long CoT
要求格式正确，答案可解析，答案匹配gold
```bash
python data/filter_long_cot.py
```

Writes:

```text
data/cold_start/teacher_long_gsm8k_train.filtered.jsonl
data/cold_start/teacher_long_gsm8k_train.rejects.jsonl
```

## 4. Estimate Difficulty
按cot长度区分easy/medium/hard
```bash
python data/estimate_difficulty.py
```

Writes:

```text
data/cold_start/teacher_long_gsm8k_train.difficulty.jsonl
```

## 5. Compress Teacher Rationales
按difficulty压缩long rationale

Recommended vLLM version:

```bash
python data/compress_teacher_rationales_vllm.py
```

Baseline Transformers version:

```bash
python data/compress_teacher_rationales.py
```

Writes:

```text
data/cold_start/teacher_compressed_gsm8k_train.jsonl
```

## 6. Filter Compressed Rationales
同filter long cot，且必须更短。但不保证质量更高
```bash
python data/filtering.py
```

Writes:

```text
data/cold_start/cold_start_gsm8k_train.jsonl
data/cold_start/teacher_compressed_gsm8k_train.rejects.jsonl
data/cold_start/compression_stats.jsonl
```

"""Ablation vLLM evaluation entrypoint.

Run from the project root:
python ablation/eval/evaluate_vllm.py

This reuses eval/evaluate_vllm.py, disables plotting, and writes all outputs
under ablation/output/eval_results/.
"""

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT_DIR / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import evaluate_vllm as eval_vllm  # noqa: E402


ABLATION_ROOT = ROOT_DIR / "ablation"
ABLATION_OUTPUT = ABLATION_ROOT / "output"

eval_vllm.OUTPUT_DIR = ABLATION_OUTPUT / "eval_results"
eval_vllm.RUN_PLOT_SCRIPT = False

eval_vllm.CHECKPOINTS = [
    {
        "name": "cold_start",
        "checkpoint": ROOT_DIR / "outputs" / "cold_start" / "student_cold_start",
        "latest_file": ROOT_DIR / "outputs" / "cold_start" / "latest_checkpoint.txt",
    },
    {
        "name": "ablation_on_policy",
        "checkpoint": ABLATION_OUTPUT / "on_policy" / "student_on_policy",
        "latest_file": ABLATION_OUTPUT / "on_policy" / "latest_checkpoint.txt",
    },
]


if __name__ == "__main__":
    eval_vllm.main()

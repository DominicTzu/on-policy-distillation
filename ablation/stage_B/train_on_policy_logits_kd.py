"""Ablation Stage B training entrypoint.

Run from the project root:
python ablation/stage_B/train_on_policy_logits_kd.py

This reuses the main Stage B implementation but writes all outputs under
ablation/output/.
"""

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
STAGE_B_DIR = ROOT_DIR / "stage_B"
if str(STAGE_B_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_B_DIR))

import train_on_policy_logits_kd as stage_b  # noqa: E402


ABLATION_ROOT = ROOT_DIR / "ablation"
ABLATION_OUTPUT = ABLATION_ROOT / "output"

# Inputs: reuse the rollout file produced by the main Stage B rollout step.
stage_b.ROLLOUT_FILE = (
    ROOT_DIR
    / "outputs"
    / "on_policy"
    / "rollouts"
    / "student_rollouts_gsm8k_train.jsonl"
)
stage_b.COLD_START_CHECKPOINT = ROOT_DIR / "outputs" / "cold_start" / "student_cold_start"
stage_b.LATEST_COLD_START_CHECKPOINT = (
    ROOT_DIR / "outputs" / "cold_start" / "latest_checkpoint.txt"
)

# Outputs: keep ablation artifacts isolated from the main run.
stage_b.OUTPUT_DIR = ABLATION_OUTPUT / "on_policy" / "student_on_policy"
stage_b.STAGE_B_MANIFEST = ABLATION_OUTPUT / "on_policy" / "stage_b_manifest.json"
stage_b.LATEST_ON_POLICY_CHECKPOINT = (
    ABLATION_OUTPUT / "on_policy" / "latest_checkpoint.txt"
)

# Ablation knobs.
stage_b.ALPHA_CE = 1.0
stage_b.BETA_KD = 0.2
stage_b.TEMPERATURE = 2.0
stage_b.EXCLUDE_EOS_FROM_KD = True

stage_b.TRAIN_ONLY_CORRECT_ROLLOUTS = True
stage_b.TRAIN_ONLY_FORMAT_VALID_ROLLOUTS = True
stage_b.MAX_ROLLOUT_RESPONSE_TOKENS = 128
stage_b.MAX_ROLLOUT_RATIONALE_TOKENS = 96


if __name__ == "__main__":
    stage_b.main()

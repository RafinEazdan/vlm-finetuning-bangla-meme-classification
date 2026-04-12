"""
Central configuration for the VLM fine-tuning pipeline.
All paths, model settings, LoRA hyperparameters, and training defaults live here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/kaggle/input/datasets/eazdanmostafarafin/bangla-meme-classification-dataset/preprocessed")
IMAGE_DIR = DATA_DIR / "Image_processed"
CSV_PATH = DATA_DIR / "Train_processed.csv"
OUTPUT_DIR = Path("/kaggle/working/finetune_output")
ADAPTER_DIR = OUTPUT_DIR / "final_adapter"
MERGED_DIR = OUTPUT_DIR / "final_merged_16bit"
RESULTS_DIR = OUTPUT_DIR / "eval"
LOGS_DIR = OUTPUT_DIR / "logs"

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit"
LOAD_IN_4BIT = True

# ── LoRA ───────────────────────────────────────────────────────────────────────
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_BIAS = "none"
FINETUNE_VISION_LAYERS = True
FINETUNE_LANGUAGE_LAYERS = True
FINETUNE_ATTENTION_MODULES = True
FINETUNE_MLP_MODULES = True

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_STEPS = 5
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "linear"
OPTIMIZER = "adamw_8bit"
MAX_SEQ_LENGTH = 4096
LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3
SEED = 42

# ── Data split ─────────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
VAL_TEST_SPLIT = 0.5
SPLIT_RANDOM_STATE = 42

# ── Inference ──────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 16


def build_instruction(labels: list[str]) -> str:
    """Build the classification instruction prompt from a list of labels."""
    return (
        "You are an image classifier. Look at the image and classify it into exactly "
        f"one of these categories: {', '.join(labels)}. "
        "Respond with only the single category name."
    )

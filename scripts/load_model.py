"""
Load a fine-tuned Qwen2.5-VL model from a saved LoRA adapter.

Can be used as a standalone script to verify the model loads correctly,
or imported by other scripts (e.g. evaluate.py).

Usage:
    python load_model.py [--adapter-dir ADAPTER_DIR]
"""

import argparse
import json
from pathlib import Path

from unsloth import FastVisionModel

import scripts.config as cfg


def load_finetuned_model(adapter_dir: str = None, load_in_4bit: bool = None):
    """Load the base model with the fine-tuned LoRA adapter applied."""
    if adapter_dir is None:
        adapter_dir = str(cfg.ADAPTER_DIR)
    if load_in_4bit is None:
        load_in_4bit = cfg.LOAD_IN_4BIT

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=adapter_dir,
        load_in_4bit=load_in_4bit,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


def load_train_metadata(output_dir: str = None):
    """Load training metadata (labels, instruction, data_dir) saved during training."""
    if output_dir is None:
        output_dir = str(cfg.OUTPUT_DIR)
    metadata_path = Path(output_dir) / "train_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Load and verify a fine-tuned model")
    parser.add_argument("--adapter-dir", type=str, default=str(cfg.ADAPTER_DIR),
                        help="Path to the saved LoRA adapter directory")
    parser.add_argument("--output-dir", type=str, default=str(cfg.OUTPUT_DIR),
                        help="Path to training output directory (for metadata)")
    args = parser.parse_args()

    print(f"Loading model from: {args.adapter_dir}")
    model, tokenizer = load_finetuned_model(args.adapter_dir)
    print(f"Model loaded successfully. Device: {model.device}")
    print(f"Model type: {type(model).__name__}")

    metadata_path = Path(args.output_dir) / "train_metadata.json"
    if metadata_path.exists():
        metadata = load_train_metadata(args.output_dir)
        print(f"Labels: {metadata['labels']}")
        print(f"Instruction: {metadata['instruction'][:80]}...")
    else:
        print(f"No training metadata found at {metadata_path}")


if __name__ == "__main__":
    main()

"""
Evaluate a fine-tuned Qwen2.5-VL model on the test split.

Usage:
    python evaluate.py [--adapter-dir ADAPTER_DIR] [--output-dir OUTPUT_DIR] [--data-dir DATA_DIR]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import scripts.config as cfg
from scripts.load_model import load_finetuned_model, load_train_metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on test set")
    parser.add_argument("--adapter-dir", type=str, default=str(cfg.ADAPTER_DIR),
                        help="Path to the saved LoRA adapter directory")
    parser.add_argument("--output-dir", type=str, default=str(cfg.OUTPUT_DIR),
                        help="Path to training output directory (for metadata and test split)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (default: use value from train_metadata.json)")
    return parser.parse_args()


def predict_label(model, tokenizer, image, instruction):
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]},
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        images=image,
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def normalize_to_label(pred_text, labels):
    low = pred_text.lower()
    for lab in labels:
        if lab.lower() in low:
            return lab
    return pred_text


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    results_dir = output_dir / "eval"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata from training
    metadata = load_train_metadata(args.output_dir)
    labels = metadata["labels"]
    instruction = metadata["instruction"]
    data_dir = Path(args.data_dir) if args.data_dir else Path(metadata["data_dir"])
    image_dir = data_dir / "Image_processed"

    # Load test split
    test_csv = output_dir / "test_split.csv"
    if not test_csv.exists():
        raise FileNotFoundError(
            f"Test split not found at {test_csv}. "
            "Run train.py first to generate the test split."
        )
    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")
    print(f"Labels: {labels}")

    # Load model
    print(f"Loading model from: {args.adapter_dir}")
    model, tokenizer = load_finetuned_model(args.adapter_dir)

    # Run predictions
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    prediction_rows = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        image = Image.open(image_dir / row["Image_name"]).convert("RGB")
        raw_pred = predict_label(model, tokenizer, image, instruction)
        pred = normalize_to_label(raw_pred, labels)
        true = row["Target"]
        is_correct = int(pred == true)

        per_class_total[true] += 1
        total += 1
        if is_correct:
            correct += 1
            per_class_correct[true] += 1

        prediction_rows.append({
            "Image_name": row["Image_name"],
            "Target": true,
            "Prediction": pred,
            "Raw_Prediction": raw_pred,
            "Correct": is_correct,
        })

    # Compute metrics
    overall_acc = correct / total if total else 0.0
    per_class_metrics = {}
    for lab in labels:
        n = per_class_total[lab]
        acc = per_class_correct[lab] / n if n else 0.0
        per_class_metrics[lab] = {
            "correct": per_class_correct[lab],
            "total": n,
            "accuracy": acc,
        }

    # Save results
    predictions_path = results_dir / "test_predictions.csv"
    metrics_path = results_dir / "test_metrics.json"

    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": {"correct": correct, "total": total, "accuracy": overall_acc},
                "per_class": per_class_metrics,
            },
            f,
            indent=2,
        )

    # Print results
    print(f"\nTest accuracy: {correct}/{total} = {overall_acc:.4f}")
    for lab in labels:
        m = per_class_metrics[lab]
        print(f"  {lab:10s}: {m['correct']}/{m['total']} = {m['accuracy']:.4f}")

    print(f"\nSaved test predictions to: {predictions_path.resolve()}")
    print(f"Saved test metrics to: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()

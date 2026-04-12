"""
Train a Qwen2.5-VL model with LoRA on the Bangla Meme Classification dataset.

Usage:
    python train.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR] [--epochs N]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL for image classification")
    parser.add_argument("--data-dir", type=str, default=str(cfg.DATA_DIR),
                        help="Path to preprocessed dataset directory")
    parser.add_argument("--output-dir", type=str, default=str(cfg.OUTPUT_DIR),
                        help="Directory to save training outputs")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=cfg.GRADIENT_ACCUMULATION_STEPS,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-seq-length", type=int, default=cfg.MAX_SEQ_LENGTH, help="Max sequence length")
    parser.add_argument("--save-merged", action="store_true", help="Also save merged 16-bit model")
    return parser.parse_args()


def load_and_split_data(data_dir: Path):
    image_dir = data_dir / "Image_processed"
    csv_path = data_dir / "Train_processed.csv"

    df = pd.read_csv(csv_path)
    df = df[["Image_name", "Target"]].dropna()
    df = df[df["Image_name"].apply(lambda n: (image_dir / n).is_file())].reset_index(drop=True)

    labels = sorted(df["Target"].unique().tolist())
    print(f"Classes: {labels}")
    print(f"Total usable samples: {len(df)}")

    train_df, temp_df = train_test_split(
        df, test_size=cfg.TEST_SIZE, stratify=df["Target"], random_state=cfg.SPLIT_RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=cfg.VAL_TEST_SPLIT, stratify=temp_df["Target"], random_state=cfg.SPLIT_RANDOM_STATE,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    return train_df, val_df, test_df, labels, image_dir


class ConversationDataset(Dataset):
    """Lazily loads images on __getitem__ to avoid holding all images in memory."""

    def __init__(self, df: pd.DataFrame, image_dir: Path, instruction: str):
        self.image_names = df["Image_name"].tolist()
        self.targets = df["Target"].astype(str).tolist()
        self.image_dir = image_dir
        self.instruction = instruction

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image = Image.open(self.image_dir / self.image_names[idx]).convert("RGB")
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": self.targets[idx]},
                    ],
                },
            ]
        }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load base model ---
    print("Loading base model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        cfg.MODEL_NAME,
        load_in_4bit=cfg.LOAD_IN_4BIT,
    )

    # --- Prepare data ---
    train_df, val_df, test_df, labels, image_dir = load_and_split_data(data_dir)
    instruction = cfg.build_instruction(labels)

    # Save metadata for evaluation later
    metadata = {
        "labels": labels,
        "instruction": instruction,
        "data_dir": str(data_dir),
    }
    with (output_dir / "train_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save test split for evaluation
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    train_dataset = ConversationDataset(train_df, image_dir, instruction)
    val_dataset = ConversationDataset(val_df, image_dir, instruction)
    print(f"Train convos: {len(train_dataset)} | Val convos: {len(val_dataset)}")

    # --- Apply LoRA ---
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.FINETUNE_VISION_LAYERS,
        finetune_language_layers=cfg.FINETUNE_LANGUAGE_LAYERS,
        finetune_attention_modules=cfg.FINETUNE_ATTENTION_MODULES,
        finetune_mlp_modules=cfg.FINETUNE_MLP_MODULES,
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        bias=cfg.LORA_BIAS,
        random_state=cfg.SEED,
        use_rslora=False,
        loftq_config=None,
    )

    # --- Configure trainer ---
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=cfg.WARMUP_STEPS,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=cfg.LOGGING_STEPS,
            optim=cfg.OPTIMIZER,
            weight_decay=cfg.WEIGHT_DECAY,
            lr_scheduler_type=cfg.LR_SCHEDULER,
            seed=cfg.SEED,
            output_dir=str(output_dir),
            logging_dir=str(output_dir / "logs"),
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=2,
            max_seq_length=args.max_seq_length,
            eval_strategy="epoch",
            save_strategy="steps",
            save_steps=cfg.SAVE_STEPS,
            save_total_limit=cfg.SAVE_TOTAL_LIMIT,
        ),
    )

    # --- Train ---
    adapter_dir = output_dir / "final_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_metrics = {}
    eval_metrics = {}
    training_completed = False

    try:
        trainer_stats = trainer.train()
        print(trainer_stats)
        train_metrics = trainer_stats.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        training_completed = True
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state...")
    finally:
        trainer.save_state()
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"Saved LoRA adapter + tokenizer to: {adapter_dir.resolve()}")

    if training_completed:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    else:
        interrupt_info = {
            "status": "interrupted",
            "global_step": int(trainer.state.global_step),
            "epoch": float(trainer.state.epoch) if trainer.state.epoch is not None else None,
        }
        with (output_dir / "interrupt_summary.json").open("w", encoding="utf-8") as f:
            json.dump(interrupt_info, f, indent=2)
        print(f"Saved interruption summary to: {(output_dir / 'interrupt_summary.json').resolve()}")

    summary = {
        "train": train_metrics,
        "eval": eval_metrics,
        "training_completed": training_completed,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved trainer artifacts under: {output_dir.resolve()}")

    # --- Optionally save merged model ---
    if args.save_merged:
        try:
            merged_dir = output_dir / "final_merged_16bit"
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
            print(f"Saved merged full model to: {merged_dir.resolve()}")
        except Exception as e:
            print("Merged full-model export was skipped:", e)


if __name__ == "__main__":
    main()

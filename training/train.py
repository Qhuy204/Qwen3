"""
Training Pipeline: Fine-tune Qwen3-VL-2B with Unsloth + LoRA.

Usage:
    python training/train.py --config configs/model_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/model_config.yaml") -> None:
    """Run the full fine-tuning pipeline."""
    config = load_config(config_path)
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]

    # ─── Step 1: Load Model ───────────────────────────────────────────
    print("=" * 60)
    print("🚀 Step 1: Loading model...")
    print(f"   Model: {model_cfg['name']}")
    print("=" * 60)

    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_cfg["name"],
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        use_gradient_checkpointing=model_cfg.get("use_gradient_checkpointing", "unsloth"),
    )

    # ─── Step 2: Apply LoRA ───────────────────────────────────────────
    print("\n🔧 Step 2: Applying LoRA adapter...")
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0),
        bias=lora_cfg.get("bias", "none"),
        random_state=lora_cfg.get("random_state", 3407),
        use_rslora=lora_cfg.get("use_rslora", False),
        loftq_config=None,
        finetune_vision_layers=lora_cfg.get("finetune_vision_layers", True),
        finetune_language_layers=lora_cfg.get("finetune_language_layers", True),
        finetune_attention_modules=lora_cfg.get("finetune_attention_modules", True),
        finetune_mlp_modules=lora_cfg.get("finetune_mlp_modules", True),
    )
    model.print_trainable_parameters()

    # ─── Step 3: Load Processed Dataset ─────────────────────────────
    print("\n📦 Step 3: Loading processed dataset...")
    from data.dataset import load_processed_dataset

    datasets = load_processed_dataset(config_path)
    train_set = datasets["train"]
    val_set = datasets.get("val")
    print(f"   Train: {len(train_set)} | Val: {len(val_set) if val_set else 0}")

    # ─── Step 4: Setup Trainer ────────────────────────────────────────
    print("\n⚙️ Step 4: Setting up trainer...")
    from trl import SFTConfig, SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator

    FastVisionModel.for_training(model)

    output_dir = train_cfg.get("output_dir", "outputs")

    # Support cả max_steps và num_train_epochs
    max_steps = train_cfg.get("max_steps", -1)
    num_train_epochs = train_cfg.get("num_train_epochs", 1)

    sft_config = SFTConfig(
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 2),
        warmup_steps=train_cfg.get("warmup_steps", 20),
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        logging_steps=train_cfg.get("logging_steps", 5),
        save_steps=train_cfg.get("save_steps", 100),
        eval_steps=train_cfg.get("eval_steps", 100),
        eval_strategy="steps" if val_set is not None else "no",
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        seed=train_cfg.get("seed", 3407),
        optim=train_cfg.get("optim", "adamw_8bit"),
        output_dir=output_dir,
        report_to="none",
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        # Vision fine-tuning required settings
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=train_cfg.get("max_length", 2048),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_set,
        eval_dataset=val_set,
        args=sft_config,
    )

    # ─── Step 5: GPU Info ─────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        reserved_gb = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        total_gb = round(gpu_stats.total_memory / 1024**3, 2)
        print(f"\n🖥️ GPU: {gpu_stats.name} | Total: {total_gb} GB | Reserved: {reserved_gb} GB")

    # ─── Step 6: Train ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🏋️ Step 6: Starting training...")
    print("=" * 60)

    trainer_stats = trainer.train()

    print("\n✅ Training complete!")
    print(f"   Total steps: {trainer_stats.global_step}")
    print(f"   Final loss:  {trainer_stats.training_loss:.4f}")

    # ─── Step 7: Save LoRA adapter ────────────────────────────────────
    save_path = Path(output_dir) / "final_lora"
    print(f"\n💾 Saving LoRA adapter to: {save_path}")
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print("✅ Saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL-2B")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args.config)

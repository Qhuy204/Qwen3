"""
Data Preparation: Download, process, and save dataset to disk.

Chạy 1 lần TRƯỚC KHI train. Dataset sẽ được lưu vào data/processed/
để train.py load nhanh mà không cần xử lý lại.

Usage:
    python data/prepare_data.py --config configs/model_config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from datasets import Dataset, load_dataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_conversations(conversations_str: str) -> list[dict[str, str]]:
    """Parse conversations JSON string → list of {role, content}."""
    if isinstance(conversations_str, list):
        return conversations_str
    return json.loads(conversations_str)


def _convert_sample_to_messages(
    sample: dict[str, Any],
    max_qa_per_image: int = 0,
) -> list[dict[str, Any]]:
    """
    Convert a single HF dataset sample → list of Unsloth message dicts.

    Each QA pair becomes one training sample with format:
    [
        {"role": "user", "content": [{"type": "text", ...}, {"type": "image", ...}]},
        {"role": "assistant", "content": [{"type": "text", ...}]}
    ]
    """
    conversations = _parse_conversations(sample["conversations"])
    image = sample["image"]

    results: list[dict[str, Any]] = []
    qa_count = 0

    for i in range(0, len(conversations) - 1, 2):
        if max_qa_per_image > 0 and qa_count >= max_qa_per_image:
            break

        user_turn = conversations[i]
        assistant_turn = conversations[i + 1]

        if user_turn.get("role") != "user" or assistant_turn.get("role") != "assistant":
            continue

        messages = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_turn["content"]},
                        {"type": "image", "image": image},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_turn["content"]},
                    ],
                },
            ]
        }
        results.append(messages)
        qa_count += 1

    return results


def prepare_and_save(config_path: str | Path) -> dict[str, int]:
    """
    Download dataset từ HuggingFace, convert sang messages format,
    split train/val/test, và save vào disk.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Dict với số lượng samples mỗi split.
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    dataset_name: str = data_cfg["dataset_name"]
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    max_qa_per_image: int = data_cfg.get("max_qa_per_image", 0)
    train_ratio: float = data_cfg.get("train_ratio", 0.85)
    val_ratio: float = data_cfg.get("val_ratio", 0.10)
    seed: int = data_cfg.get("seed", 42)

    processed_dir.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Download từ HuggingFace ──────────────────────────────
    print("=" * 60)
    print("📦 Step 1: Downloading dataset từ HuggingFace...")
    print(f"   Dataset: {dataset_name}")
    print("=" * 60)

    raw_dataset = load_dataset(dataset_name, split="train")
    print(f"   ✅ {len(raw_dataset)} images loaded")

    # ─── Step 2: Convert sang messages format ─────────────────────────
    qa_label = "tất cả" if max_qa_per_image == 0 else f"max {max_qa_per_image}"
    print(f"\n🔄 Step 2: Converting to messages format ({qa_label} QA/image)...")

    all_samples: list[dict[str, Any]] = []
    for idx, sample in enumerate(raw_dataset):
        if idx % 5000 == 0:
            print(f"   Processing image {idx}/{len(raw_dataset)}...")
        qa_samples = _convert_sample_to_messages(sample, max_qa_per_image)
        all_samples.extend(qa_samples)

    print(f"   ✅ {len(all_samples)} total QA samples")

    # ─── Step 3: Shuffle & Split ──────────────────────────────────────
    print(f"\n🔀 Step 3: Shuffling & splitting (seed={seed})...")

    random.seed(seed)
    random.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train : n_train + n_val],
        "test": all_samples[n_train + n_val :],
    }

    # ─── Step 4: Save to disk ─────────────────────────────────────────
    print(f"\n💾 Step 4: Saving to {processed_dir}/")

    result_counts: dict[str, int] = {}
    for split_name, split_data in splits.items():
        ds = Dataset.from_list(split_data)
        save_path = processed_dir / split_name
        ds.save_to_disk(str(save_path))
        result_counts[split_name] = len(ds)
        print(f"   ✅ {split_name}: {len(ds)} samples → {save_path}")

    print(f"\n{'=' * 60}")
    print("✅ Data preparation complete!")
    print(f"   Train: {result_counts['train']}")
    print(f"   Val:   {result_counts['val']}")
    print(f"   Test:  {result_counts['test']}")
    print(f"   Saved to: {processed_dir.resolve()}")
    print(f"\n   👉 Bây giờ chạy: python training/train.py --config {config_path}")
    print("=" * 60)

    return result_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare VQA dataset for training")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    prepare_and_save(args.config)

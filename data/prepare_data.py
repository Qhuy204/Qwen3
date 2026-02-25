"""
Data Preparation: Download, process, and save dataset to disk.
Sử dụng generator để tránh tràn RAM hệ thống khi xử lý 1.1M samples.

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
    """Convert sample → list of messages."""
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

        results.append({
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
                    "content": [{"type": "text", "text": assistant_turn["content"]}],
                },
            ]
        })
        qa_count += 1
    return results


def prepare_and_save(config_path: str | Path) -> None:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    dataset_name = data_cfg["dataset_name"]
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    max_qa_per_image = data_cfg.get("max_qa_per_image", 0)
    train_ratio = data_cfg.get("train_ratio", 0.85)
    val_ratio = data_cfg.get("val_ratio", 0.10)
    seed = data_cfg.get("seed", 42)

    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"📦 Loading dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name, split="train")
    print(f"✅ {len(raw_dataset)} images loaded")

    # ─── Step 2: Generator processing (RAM efficient) ────────────────
    print(f"\n🔄 Processing all QA pairs (max_qa={max_qa_per_image})...")
    
    def gen_samples():
        for sample in raw_dataset:
            yield from _convert_sample_to_messages(sample, max_qa_per_image)

    # Chuyển generator thành Dataset (Streaming to disk)
    full_dataset = Dataset.from_generator(gen_samples)
    print(f"✅ Total QA samples: {len(full_dataset)}")

    # ─── Step 3: Shuffle & Split ──────────────────────────────────────
    print(f"🔀 Shuffling and splitting...")
    full_dataset = full_dataset.shuffle(seed=seed)
    
    # Split
    ds_split = full_dataset.train_test_split(test_size=(1 - train_ratio), seed=seed)
    train_ds = ds_split["train"]
    
    # Tiếp tục split test ra val/test
    remaining_ratio = val_ratio / (1 - train_ratio)
    val_test_split = ds_split["test"].train_test_split(test_size=(1 - remaining_ratio), seed=seed)
    
    val_ds = val_test_split["train"]
    test_ds = val_test_split["test"]

    # ─── Step 4: Save ────────────────────────────────────────────────
    print(f"\n💾 Saving to {processed_dir}/")
    train_ds.save_to_disk(str(processed_dir / "train"))
    val_ds.save_to_disk(str(processed_dir / "val"))
    test_ds.save_to_disk(str(processed_dir / "test"))

    print(f"\n✅ Xong! Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"👉 Bây giờ bạn có thể chạy: python training/train.py --config {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    prepare_and_save(args.config)

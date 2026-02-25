"""
Data Preparation In-Memory: Tận dụng 167GB RAM để xử lý siêu tốc.
Bỏ qua Multiprocessing để tránh nghẽn I/O, xử lý trực tiếp trên RAM.

Usage:
    python data/prepare_data.py --config configs/model_config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_conversations(conversations_str: str) -> list[dict[str, str]]:
    if isinstance(conversations_str, list) or conversations_str is None:
        return conversations_str
    return json.loads(conversations_str)


def prepare_and_save(config_path: str | Path) -> None:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    dataset_name = data_cfg["dataset_name"]
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    max_qa_limit = data_cfg.get("max_qa_per_image", 5) # Mặc định gộp 5 QA vào 1 hội thoại
    train_ratio = data_cfg.get("train_ratio", 0.85)
    seed = data_cfg.get("seed", 42)

    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🚀 In-Memory High Speed Preparation")
    print(f"📦 Loading dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name, split="train")
    print(f"✅ {len(raw_dataset)} images loaded into RAM")

    # ─── Step 1: In-Memory Processing ──────────────────────────────
    print(f"\n🔄 Processing and Multi-turn grouping (Single-thread RAM mode)...")
    
    all_samples = []
    total_images = len(raw_dataset)
    
    # Gộp 5 câu hỏi vào 1 turn cho đỡ nặng context
    limit = 5 if max_qa_limit == 0 else max_qa_limit

    for idx, item in enumerate(raw_dataset):
        if idx % 1000 == 0:
            print(f"   Progress: {idx}/{total_images} images processed...")
            
        convs = _parse_conversations(item["conversations"])
        if not convs: continue
        
        img = item["image"]
        current_msgs = []
        qa_counter = 0

        for j in range(0, len(convs) - 1, 2):
            user_turn = convs[j]
            assistant_turn = convs[j + 1]

            if len(current_msgs) == 0:
                user_content = [
                    {"type": "text", "text": user_turn["content"]},
                    {"type": "image", "image": img},
                ]
            else:
                user_content = [{"type": "text", "text": user_turn["content"]}]

            current_msgs.append({"role": "user", "content": user_content})
            current_msgs.append({
                "role": "assistant", 
                "content": [{"type": "text", "text": assistant_turn["content"]}]
            })
            
            qa_counter += 1
            if qa_counter >= limit:
                all_samples.append({"messages": current_msgs})
                current_msgs = []
                qa_counter = 0

        if current_msgs:
            all_samples.append({"messages": current_msgs})

    # ─── Step 2: Create Dataset ──────────────────────────────────────
    print(f"\n🏁 Step 2: Creating final dataset from {len(all_samples)} samples...")
    full_dataset = Dataset.from_list(all_samples)

    print(f"🔀 Step 3: Shuffling and splitting...")
    full_dataset = full_dataset.shuffle(seed=seed)
    ds_split = full_dataset.train_test_split(test_size=(1 - train_ratio), seed=seed)

    # ─── Step 4: Save ────────────────────────────────────────────────
    print(f"\n💾 Step 4: Saving to disk (Sequential write)...")
    ds_split["train"].save_to_disk(str(processed_dir / "train"))
    ds_split["test"].save_to_disk(str(processed_dir / "val"))

    print(f"\n{'=' * 60}")
    print("✅ Xong! Dataset chuẩn bị cực nhanh bằng RAM mode.")
    print(f"   Train: {len(ds_split['train'])} samples")
    print(f"   Val:   {len(ds_split['test'])} samples")
    print(f"👉 Chạy training: python training/train.py {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_pos", type=str, nargs="?", help="Path to YAML config")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    config_path = args.config_pos if args.config_pos else args.config
    prepare_and_save(config_path)

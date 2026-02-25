"""
Data Preparation High-Performance: Xử lý song song (Parallel Processing).
Tận dụng tối đa RAM 167GB và đa nhân CPU để chuẩn bị dữ liệu cực nhanh.

Usage:
    python data/prepare_data.py --config configs/model_config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_conversations(conversations_str: str) -> list[dict[str, str]]:
    if isinstance(conversations_str, list):
        return conversations_str
    return json.loads(conversations_str)


def _process_map_fn(item: dict[str, Any], max_qa: int = 5) -> dict[str, Any]:
    """Hàm map để xử lý song song từng hàng."""
    conversations = _parse_conversations(item["conversations"])
    image = item["image"]
    
    results = []
    current_messages = []
    qa_count = 0

    for i in range(0, len(conversations) - 1, 2):
        user_turn = conversations[i]
        assistant_turn = conversations[i + 1]

        if len(current_messages) == 0:
            user_content = [
                {"type": "text", "text": user_turn["content"]},
                {"type": "image", "image": image},
            ]
        else:
            user_content = [{"type": "text", "text": user_turn["content"]}]

        current_messages.append({"role": "user", "content": user_content})
        current_messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_turn["content"]}]})
        
        qa_count += 1
        if qa_count >= max_qa:
            results.append(current_messages)
            current_messages = []
            qa_count = 0

    if current_messages:
        results.append(current_messages)
    
    # Dataset.map yêu cầu trả về dictionary với các cột mới
    return {"all_messages": results}


def prepare_and_save(config_path: str | Path) -> None:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    dataset_name = data_cfg["dataset_name"]
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    max_qa_limit = data_cfg.get("max_qa_per_image", 0)
    train_ratio = data_cfg.get("train_ratio", 0.85)
    seed = data_cfg.get("seed", 42)

    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"� High-Performance Data Preparation")
    print(f"📦 Loading dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name, split="train")
    
    # Tự động lấy số CPU core
    num_cpus = os.cpu_count() or 4
    print(f"⚙️ Using {num_cpus} CPU cores for parallel processing...")

    # ─── Step 1: Parallel Map ────────────────────────────────────────
    limit = 5 if max_qa_limit == 0 else min(5, max_qa_limit)
    
    print(f"🔄 Processing and grouping (Parallel)...")
    processed_raw = raw_dataset.map(
        _process_map_fn,
        fn_kwargs={"max_qa": limit},
        num_proc=num_cpus,
        remove_columns=raw_dataset.column_names,
        desc="Parallel Processing"
    )

    # ─── Step 2: Flatten (Vì 1 ảnh ra nhiều sample) ──────────────────
    print(f"💥 Flattening dataset...")
    
    def flatten_gen():
        for item in processed_raw:
            for msg_list in item["all_messages"]:
                yield {"messages": msg_list}

    full_dataset = Dataset.from_generator(flatten_gen)
    print(f"✅ Total samples: {len(full_dataset)}")

    # ─── Step 3: Shuffle & Split ──────────────────────────────────────
    print(f"🔀 Shuffling and splitting...")
    full_dataset = full_dataset.shuffle(seed=seed)
    ds_split = full_dataset.train_test_split(test_size=(1 - train_ratio), seed=seed)

    # ─── Step 4: Save ────────────────────────────────────────────────
    print(f"\n💾 Saving to disk (Fast I/O)...")
    ds_split["train"].save_to_disk(str(processed_dir / "train"))
    ds_split["test"].save_to_disk(str(processed_dir / "val"))

    print(f"\n✅ Done! Parallel processing finished successfully.")
    print(f"👉 Run training: python training/train.py {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_pos", type=str, nargs="?", help="Path to YAML config")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    config_path = args.config_pos if args.config_pos else args.config
    prepare_and_save(config_path)
